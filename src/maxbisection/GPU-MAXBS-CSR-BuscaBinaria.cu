#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cub/cub.cuh"

//#define INF (LLONG_MIN/2)
#define INF (-9999999999.0)

// Check for errors in the use of CUDA API functions in C/C++
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Performs an atomic addition of a double value at a memory address on the GPU
static inline __device__ float atomicAdd2(float *address, float val) {
    int *address_as_int = (int *)address;
    int old = *address_as_int, assumed;

    do {
        assumed = old;
        old = atomicCAS(
            address_as_int,
            assumed,
            __float_as_int(val + __int_as_float(assumed))
        );
    } while (assumed != old);

    return __int_as_float(old);
}

using namespace std;
using ll = long long;
using Clock = chrono::steady_clock;

static Clock::time_point g_start;
static double g_timeLimit; // segundos
static double elapsed() {
    return chrono::duration<double>(Clock::now() - g_start).count();
}
static bool timeUp() { 
    return elapsed() >= g_timeLimit; 
}
static mt19937 rng;
static int randInt(int lo, int hi) { 
    uniform_int_distribution<int> d(lo, hi);
    return d(rng);
}

struct Solution {
    vector<int> part;
    vector<float> gain;
    float fitness = 0.0f;

    void init(int n) { 
        part.assign(n, 0); 
        gain.assign(n, 0.0f);
        fitness = 0.0f;
    }
};

struct MacutParams {
    int      p         = 10;
    int      Piter     = 500;
    int      gamma     = 150;
    ll       MaxIter   = 1000000;
    int      maxt      = 200;
    double   timeLimit = 60.0;
    unsigned seed      = (unsigned)time(nullptr);
    string   outFile;
};

int* CSR_ROWPTR;    // The rowptr vector of the CSR matrix indicates the number of neighbors for each vertex
int* CSR_COLIND;    // The colind vector of the CSR matrix stores the neighbors of each vertex.
float* CSR_VAL;    // The val vector of the CSR matrix stores the edge values between the vertex and its neighbors.

int *d_CSR_R;       // GPU CSR_ROWPTR vector
int *d_CSR_COL;     // GPU CSR_COLIND vector
float *d_CSR_VAL;  // GPU CSR_VAL vector

int *d_current_solution;
int *d_best_solution; 
int *d_tabu;
int *d_pos;
float *d_gain;
float *d_vertex_gain;
float *d_best_vertex_gain;
float *d_aux_weight;

void *d_temp_storage = NULL;
size_t temp_storage_bytes = 0;

cub::KeyValuePair <int, float> h_out;
cub::KeyValuePair <int, float> *d_out;

int total_nbr_v = 0;
int nbr_v       = 0;
int delta_nbr_v = 0;
int nbr_movs    = 0;
int nbr_edges   = 0;

ll iterations_count  = 0;
ll avaliations_count = 0;

vector<vector<pair<int,float>>> adj; 

void createCSR(){
    for(int u = 0; u < total_nbr_v; u++){
        std::sort(adj[u].begin(), adj[u].end(),
            [](const pair<int,float>& a, const pair<int,float>& b){
                return a.first < b.first;
            });

        int pos = CSR_ROWPTR[u];

        for(const auto &e : adj[u]){
            CSR_COLIND[pos] = e.first;
            CSR_VAL[pos]    = e.second;
            pos++;
        }
    }
    
    cudaMalloc(&d_CSR_COL, 2 * nbr_edges * sizeof(int));
    cudaMalloc(&d_CSR_VAL, 2 * nbr_edges * sizeof(float));
    cudaMalloc(&d_CSR_R,   (total_nbr_v + 1) * sizeof(int));
    
    cudaMemcpy(d_CSR_COL, CSR_COLIND, 2 * nbr_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_CSR_VAL, CSR_VAL,    2 * nbr_edges * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_CSR_R,   CSR_ROWPTR, (total_nbr_v + 1) * sizeof(int), cudaMemcpyHostToDevice);
}

void load(const string &path) {
    ifstream fin(path);
    if (!fin) {
        cerr << "Erro: nao foi possivel abrir o arquivo do grafo: " << path << "\n";
        exit(1);
    }

    fin >> total_nbr_v >> total_nbr_v >> nbr_edges;

    CSR_ROWPTR = (int*)calloc(total_nbr_v + 1, sizeof(int));
    CSR_COLIND = (int*)malloc(nbr_edges * 2 * sizeof(int));
    CSR_VAL    = (float*)malloc(nbr_edges * 2 * sizeof(float));
    CSR_ROWPTR[0] = 0;

    nbr_v       = total_nbr_v / 2;
    delta_nbr_v = total_nbr_v - nbr_v;
    nbr_movs    = nbr_v * delta_nbr_v;

    adj.assign(total_nbr_v, {});
    for (int e = 0; e < nbr_edges; e++) {
        int u, v;
        float w;

        fin >> u >> v >> w;
        --u; --v;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
        CSR_ROWPTR[u+1] += 1;
        CSR_ROWPTR[v+1] += 1;
    }

    for(int x = 0; x < total_nbr_v; x++){
        CSR_ROWPTR[x+1] += CSR_ROWPTR[x];
    }

    createCSR();
}

static void setup_data(){
    cudaMalloc((void**)&d_current_solution, total_nbr_v * sizeof(int));
    cudaMalloc((void**)&d_best_solution,    total_nbr_v * sizeof(int));
    cudaMalloc((void**)&d_tabu,             total_nbr_v * sizeof(int));
    cudaMalloc((void**)&d_pos,              total_nbr_v * sizeof(int));
    cudaMalloc((void**)&d_gain,             nbr_movs    * sizeof(float));
    cudaMalloc((void**)&d_vertex_gain,      total_nbr_v * sizeof(float));
    cudaMalloc((void**)&d_best_vertex_gain, total_nbr_v * sizeof(float));
    cudaMalloc((void**)&d_aux_weight,       1           * sizeof(float));
    cudaMalloc((void**)&d_out,              sizeof(cub::KeyValuePair <int, float>));
}

static void randomPartition(Solution &s, int n) {
    vector<int> vertices(n);
    iota(vertices.begin(), vertices.end(), 0);
    shuffle(vertices.begin(), vertices.end(), rng);
    fill(s.part.begin(), s.part.end(), 1);
    for (int i = 0; i < nbr_v; i++) {
        s.part[vertices[i]] = 0;
    }
}

static float computeFitnessFromScratch(const Solution &s) {
    float f = 0.0f;
    for (int v = 0; v < total_nbr_v; v++)
        for (auto &pr : adj[v]) {
            int    u = pr.first;
            float  w = pr.second;
            if (u > v && s.part[u] != s.part[v]) f += w;
        }
    return f;
}

static void computeGainsFromScratch(Solution &s) {
    for (int v = 0; v < total_nbr_v; v++) {
        float same = 0.0, opp = 0.0;
        for (auto &pr : adj[v]) {
            int    u = pr.first;
            float  w = pr.second;
            if (s.part[u] == s.part[v]) same += w;
            else opp += w;
        }
        s.gain[v] = same - opp;
    }
}

static bool isValidBisection(const Solution &s) {
    int countS1 = 0;
    for(int v = 0; v < total_nbr_v; v++){
        if(s.part[v] == 0) countS1++;
    }
    return countS1 == nbr_v;
}

static void verifySolutionWeight(const Solution &s, float calculated_weight) {
    float real_weight = computeFitnessFromScratch(s);

    cout << fixed << setprecision(3);
    cout << "Peso calculado durante a tabu: " << calculated_weight << "\n";
    cout << "Peso recalculado do zero:      " << real_weight << "\n";
    if (fabs(calculated_weight - real_weight) < 1e-2f) cout << "PESO CORRETO\n";
    else cout << "ERRO NO PESO! Diferenca:       " << calculated_weight - real_weight << "\n";

    int countS1 = 0;
    for (int v = 0; v < total_nbr_v; v++) {
        if (s.part[v] == 0) countS1++;
    }
    cout << "Vertices em S1: " << countS1 << "\n";
    cout << "Vertices em S2: " << total_nbr_v - countS1 << "\n";
    if (isValidBisection(s)) cout << "BISSECTION CORRETA\n";
    else                     cout << "ERRO: PARTICAO DESBALANCEADA\n";
}

static int tabuTenure(ll iter, int maxt) {
    static const int b[15] = {1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1};
    ll period = 16LL * maxt;
    ll it = ((iter - 1) % period) + 1;
    ll pos = 1;

    for (int i = 0; i < 15; i++) {
        ll tenure = (1LL * maxt * b[i]) / 8;
        ll length = 4LL * tenure;
        if (it < pos + length)
            return (int)tenure;
        pos += length;
    }

    return maxt / 8;
}

static void applyMove(Solution &s, int v) {
    float g_v  = s.gain[v];
    s.fitness  += g_v;                
    int oldSide = s.part[v];
    for (auto &pr : adj[v]) {
        int    u = pr.first; 
        float  w  = pr.second;
        if (s.part[u] == oldSide) s.gain[u] -= 2.0f * w;
        else                      s.gain[u] += 2.0f * w;
    }
    s.part[v] = 1 - oldSide;
    s.gain[v] = -g_v;
}

static void applySwap(Solution &s, int u, int v) {
    if (s.part[u] == s.part[v]) return;
    applyMove(s, u);
    applyMove(s, v);
}

static void applyRandomSwap(Solution &s) {
    int n = (int)s.part.size();
    int u = randInt(0, n - 1);
    int v = randInt(0, n - 1);
    while (s.part[v] == s.part[u]) {
        v = randInt(0, n - 1);
    }
    applySwap(s, u, v);
}

static void currentToSolution(Solution &s, int *current_solution, float *vertex_gain, float current_weight){
    for (int i = 0; i < total_nbr_v; i++) {
        if(i < nbr_v){
            s.part[current_solution[i]] = 0;
        }else{
            s.part[current_solution[i]] = 1;
        }
        s.gain[current_solution[i]] = vertex_gain[current_solution[i]];
    }
    s.fitness = current_weight;
}

static void initialSolution(Solution &s, int *current_solution, float *vertex_gain, int *tabu, int *pos){
    int i, x = 0, y = nbr_v;

    for(i = 0; i < total_nbr_v; i++){
        tabu[i]        = 0;
        vertex_gain[i] = s.gain[i];

        if(s.part[i] == 0){
            current_solution[x] = i;
            pos[i]              = x;
            x++;
        }else{
            current_solution[y] = i;
            pos[i]              = y;
            y++;  
        }
    }
}

__device__ __forceinline__ float gpu_edge(int u, int v, const int* __restrict__ d_CSR_R, const int* __restrict__ d_CSR_COL, const float* __restrict__ d_CSR_VAL){
    int left  = d_CSR_R[u];
    int right = d_CSR_R[u + 1] - 1;

    while(left <= right){
        int mid = (left + right) >> 1;
        int z   = d_CSR_COL[mid];
        if(z == v) return d_CSR_VAL[mid];
        if(z < v) left  = mid + 1;
        else      right = mid - 1;
    }
    return 0.0f;
}

__global__ void calculateMoveGain(int *d_current_solution, float *d_gain, float *d_vertex_gain, int *d_tabu, int* d_CSR_R, int* d_CSR_COL, float* d_CSR_VAL, int total_nbr_v, int nbr_v, int delta_nbr_v, int iter, float current_weight, float best_weight, int nbr_movs){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < nbr_movs){
        int idx_u = idx/delta_nbr_v;
        int idx_v = (idx%delta_nbr_v) + nbr_v;
        int vertex_u = d_current_solution[idx_u];
        int vertex_v = d_current_solution[idx_v];
        float edge_aux = gpu_edge(vertex_u, vertex_v, d_CSR_R, d_CSR_COL, d_CSR_VAL);

        d_gain[idx] = d_vertex_gain[vertex_u] + d_vertex_gain[vertex_v] + 2.0f * edge_aux;

        if((d_tabu[vertex_u] > iter || d_tabu[vertex_v] > iter) && (current_weight + d_gain[idx] <= best_weight)){
            d_gain[idx] = INF + d_gain[idx];
        }
    }
}

__global__ void updateMoveGain(int *d_current_solution, float *d_vertex_gain, int *d_pos, int* d_CSR_R, int* d_CSR_COL, float* d_CSR_VAL, int index_u, int index_v, int nbr_v){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vertex_u = d_current_solution[index_u];
    int vertex_v = d_current_solution[index_v];
    int qnt_u = d_CSR_R[vertex_u+1] - d_CSR_R[vertex_u];
    int qnt_v = d_CSR_R[vertex_v+1] - d_CSR_R[vertex_v];

    if(idx < qnt_u){
        int   vertex_z    = d_CSR_COL[d_CSR_R[vertex_u] + idx];
        float edge_weight = d_CSR_VAL[d_CSR_R[vertex_u] + idx];
        if(d_pos[vertex_z] < nbr_v) atomicAdd2(&d_vertex_gain[vertex_z], -2.0f * edge_weight);
        else                        atomicAdd2(&d_vertex_gain[vertex_z],  2.0f * edge_weight);        
    }

    if(idx < qnt_v){
        int   vertex_z    = d_CSR_COL[d_CSR_R[vertex_v] + idx];
        float edge_weight = d_CSR_VAL[d_CSR_R[vertex_v] + idx];
        if(d_pos[vertex_z] >= nbr_v) atomicAdd2(&d_vertex_gain[vertex_z], -2.0f * edge_weight);
        else                         atomicAdd2(&d_vertex_gain[vertex_z],  2.0f * edge_weight);
    }
}

__global__ void finalizeMove(int *d_current_solution, float *d_vertex_gain, int *d_pos, int *d_tabu, float *d_aux_weight, int* d_CSR_R, int* d_CSR_COL, float* d_CSR_VAL, int index_u, int index_v, int iter, int tt, int update_tabu){
    int vertex_u = d_current_solution[index_u];
    int vertex_v = d_current_solution[index_v];

    if(!update_tabu){
        float edge_aux  = gpu_edge(vertex_u, vertex_v, d_CSR_R, d_CSR_COL, d_CSR_VAL);
        d_aux_weight[0] = d_vertex_gain[vertex_u] + d_vertex_gain[vertex_v] - 2.0f * edge_aux;
    }

    d_vertex_gain[vertex_u] = -d_vertex_gain[vertex_u];
    d_vertex_gain[vertex_v] = -d_vertex_gain[vertex_v];

    d_current_solution[index_u] = vertex_v;
    d_current_solution[index_v] = vertex_u;

    d_pos[vertex_u] = index_v;
    d_pos[vertex_v] = index_u;

    if(update_tabu){
        d_tabu[vertex_u] = iter + tt;
        d_tabu[vertex_v] = iter + tt;
    }
}

/*
v:    0 1 2 3 4 5 6
part: 1 0 0 1 1 0 1
curn: 1 2 5 0 3 4 6
*/

static void perturbationBasedTabuSearch(Solution &I, int Piter, ll MaxIter, int gamma, int maxt, Solution &best) {
    int   current_solution[total_nbr_v];
    int   best_solution[total_nbr_v];
    int   tabu[total_nbr_v];
    int   pos[total_nbr_v];
    float vertex_gain[total_nbr_v];
    float best_vertex_gain[total_nbr_v];

    float current_weight = 0.0;
    float best_weight    = 0.0;
    float aux_weight[1]  = {0.0};

    computeGainsFromScratch(I);
    I.fitness = computeFitnessFromScratch(I);
    best = I; 

    initialSolution(I, current_solution, vertex_gain, tabu, pos);
    current_weight = I.fitness;
    best_weight    = I.fitness;

    gpuErrchk(cudaMemcpy(d_current_solution, current_solution, total_nbr_v * sizeof(int),    cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_best_solution,    current_solution, total_nbr_v * sizeof(int),    cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_tabu,             tabu,             total_nbr_v * sizeof(int),    cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_pos,              pos,              total_nbr_v * sizeof(int),    cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_vertex_gain,      vertex_gain,      total_nbr_v * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_best_vertex_gain, vertex_gain,      total_nbr_v * sizeof(float), cudaMemcpyHostToDevice));

    if(temp_storage_bytes == 0){
        gpuErrchk(cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_gain, d_out, nbr_movs));
        gpuErrchk(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    }

    int g_eff            = min(gamma, total_nbr_v);
    int numPertSwaps     = max(1, g_eff / 2);
    int blocksVertexGain = total_nbr_v / 256 + 1;
    int blocksSwapGain   = nbr_movs / 256 + 1;
    //int blocksVertexPertub = g_eff / 256 + 1;

    ll iter = 0;
    ll iterSinceImprovement = 0;
    while (iter < MaxIter) {
        iterations_count++;
        avaliations_count += nbr_movs;
        //if ((iter & 4095) == 0 && timeUp()) break;
        if (timeUp()) break;

        calculateMoveGain<<<blocksSwapGain, 256>>>(d_current_solution, d_gain, d_vertex_gain, d_tabu, d_CSR_R, d_CSR_COL, d_CSR_VAL, total_nbr_v, nbr_v, delta_nbr_v, iter, current_weight, best_weight, nbr_movs);
        gpuErrchk( cudaPeekAtLastError() );

        gpuErrchk( cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_gain, d_out, nbr_movs) );
        gpuErrchk( cudaMemcpy(&h_out, d_out, sizeof(cub::KeyValuePair <int, float>), cudaMemcpyDeviceToHost) );

        int   idx   = h_out.key;
        float value = h_out.value;

        if(value < INF/4){
            current_weight += value + (-1)*INF;
        }else{
            current_weight += value;
        }

        int tt      = tabuTenure(iter, maxt);
        int index_u = idx / delta_nbr_v;
        int index_v = (idx % delta_nbr_v) + nbr_v;

        updateMoveGain<<<blocksVertexGain, 256>>>(d_current_solution, d_vertex_gain, d_pos, d_CSR_R, d_CSR_COL, d_CSR_VAL, index_u, index_v, nbr_v);
        finalizeMove<<<1,1>>>(d_current_solution, d_vertex_gain, d_pos, d_tabu, d_aux_weight, d_CSR_R, d_CSR_COL, d_CSR_VAL, index_u, index_v, iter, tt, 1);

        if(current_weight > best_weight){
            gpuErrchk( cudaMemcpy(d_best_solution,    d_current_solution, total_nbr_v * sizeof(int),   cudaMemcpyDeviceToDevice) );
            gpuErrchk( cudaMemcpy(d_best_vertex_gain, d_vertex_gain,      total_nbr_v * sizeof(float), cudaMemcpyDeviceToDevice) );
            best_weight = current_weight;
            iterSinceImprovement = 0;
        }else{
            iterSinceImprovement++;
        }

        iter++;

        if (iterSinceImprovement >= Piter) {
            unordered_set<int> used_moves;

            for (int p = 0; p < numPertSwaps; p++) {
                int move_idx;

                do { move_idx = randInt(0, nbr_movs - 1);
                } while (used_moves.count(move_idx));
                used_moves.insert(move_idx);

                index_u = move_idx / delta_nbr_v;
                index_v = (move_idx % delta_nbr_v) + nbr_v;

                updateMoveGain<<<blocksVertexGain, 256>>>(d_current_solution, d_vertex_gain, d_pos, d_CSR_R, d_CSR_COL, d_CSR_VAL, index_u, index_v, nbr_v);
                finalizeMove<<<1,1>>>(d_current_solution, d_vertex_gain, d_pos, d_tabu, d_aux_weight, d_CSR_R, d_CSR_COL, d_CSR_VAL, index_u, index_v, iter, 0, 0);

                gpuErrchk(cudaMemcpy(aux_weight, d_aux_weight, sizeof(float), cudaMemcpyDeviceToHost));
                current_weight += aux_weight[0];
            }

            iterSinceImprovement = 0;
        }
    }

    gpuErrchk( cudaMemcpy(current_solution, d_current_solution, total_nbr_v * sizeof(int),   cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(best_solution,    d_best_solution,    total_nbr_v * sizeof(int),   cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(vertex_gain,      d_vertex_gain,      total_nbr_v * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(best_vertex_gain, d_best_vertex_gain, total_nbr_v * sizeof(float), cudaMemcpyDeviceToHost) );
    
    currentToSolution(I,    current_solution, vertex_gain,      current_weight);
    currentToSolution(best, best_solution,    best_vertex_gain, best_weight);
    verifySolutionWeight(I, current_weight);
}

static bool samePartition(const Solution &a, const Solution &b) {
    bool eq = true, comp = true;
    for (size_t i = 0; i < a.part.size(); i++) {
        if (a.part[i] != b.part[i]) eq   = false;
        if (a.part[i] == b.part[i]) comp = false;
    }
    return eq || comp;
}

static vector<Solution> generatePopulation(int p, int Piter, ll MaxIter, int gamma, int maxt) {
    vector<Solution> candidates;
    int n = total_nbr_v;
    int attempts = 0;
    int maxAttempts = 20 * p + 100;

    while ((int)candidates.size() < 2 * p && attempts < maxAttempts && !timeUp()) {
        attempts++;
        Solution start; 
        start.init(n);
        randomPartition(start, n);

        Solution local; 
        local.init(n);
        perturbationBasedTabuSearch(start, Piter, MaxIter, gamma, maxt, local);

        bool dup = false;
        for (auto &c : candidates) {
            if (samePartition(c, local)) { 
                dup = true; 
                break; 
            }
        }
        if (!dup) candidates.push_back(local);
    }

    sort(candidates.begin(), candidates.end(),
         [](const Solution &a, const Solution &b) { 
            return a.fitness > b.fitness; 
        });

    int keep = min((int)candidates.size(), p);
    vector<Solution> pop(candidates.begin(), candidates.begin() + keep);

    while ((int)pop.size() < 2 && !pop.empty()) {
        Solution clone = pop[0];
        if (n > 1) {
            applyRandomSwap(clone);
        }
        pop.push_back(clone);
    }
    if (pop.empty()) {
        Solution s; 
        s.init(n);
        randomPartition(s, n);
        computeGainsFromScratch(s);
        s.fitness = computeFitnessFromScratch(s);
        pop.push_back(s);
        Solution s2 = s;
        if (n > 1) applyRandomSwap(s2);
        pop.push_back(s2);
    }

    return pop;
}

static void repairBisection(Solution &s) {
    int n = (int)s.part.size();
    int k = n / 2;
    vector<int> S1;
    vector<int> S2;

    for (int v = 0; v < n; v++) {
        if (s.part[v] == 0) S1.push_back(v);
        else                S2.push_back(v);
    }

    shuffle(S1.begin(), S1.end(), rng);
    shuffle(S2.begin(), S2.end(), rng);

    while ((int)S1.size() > k) {
        int v = S1.back();
        S1.pop_back();
        s.part[v] = 1;
        S2.push_back(v);
    }
    while ((int)S1.size() < k) {
        int v = S2.back();
        S2.pop_back();
        s.part[v] = 0;
        S1.push_back(v);
    }
}

static Solution recombination(const vector<const Solution*> &parents) {
    int n = total_nbr_v;
    int m = (int)parents.size();
    vector<char> assigned(n, 0);
    Solution offspring; offspring.init(n);

    auto buildOneSide = [&](int sideIndex) {
        int bestCount = -1;
        int bestCombo = 0;
        int numCombos = 1 << m;
        for (int combo = 0; combo < numCombos; combo++) {
            int cnt = 0;
            for (int v = 0; v < n; v++) {
                if (assigned[v]) continue;
                bool matchAll = true;
                for (int i = 0; i < m; i++) {
                    int bit = (combo >> i) & 1; // 0 ou 1 representa qual lado do pai i escolhido
                    if (parents[i]->part[v] != bit) { 
                        matchAll = false; 
                        break; 
                    }
                }
                if (matchAll) cnt++;
            }
            if (cnt > bestCount) { 
                bestCount = cnt; 
                bestCombo = combo; 
            }
        }
        
        for (int v = 0; v < n; v++) {
            if (assigned[v]) continue;
            bool matchAll = true;
            for (int i = 0; i < m; i++) {
                int bit = (bestCombo >> i) & 1;
                if (parents[i]->part[v] != bit) { 
                    matchAll = false; 
                    break; 
                }
            }
            if (matchAll) {
                offspring.part[v] = sideIndex;
                assigned[v] = 1;
            }
        }
    };

    buildOneSide(0);
    buildOneSide(1); 

    for(int v = 0; v < n; v++) 
        if(!assigned[v]) offspring.part[v] = randInt(0, 1);

    repairBisection(offspring);
    computeGainsFromScratch(offspring);
    offspring.fitness = computeFitnessFromScratch(offspring);
    return offspring;
}

static void poolUpdating(const Solution &offspring, vector<Solution> &pop) {
    for (auto &s : pop)
        if (samePartition(s, offspring)) 
            return; 
        
    int worstIdx = 0;
    for (size_t i = 1; i < pop.size(); i++)
        if (pop[i].fitness < pop[worstIdx].fitness) worstIdx = (int)i;

    if (offspring.fitness > pop[worstIdx].fitness)
        pop[worstIdx] = offspring;
}

static Solution macut(const MacutParams &params, ll &generations) {
    g_start = Clock::now();
    g_timeLimit = params.timeLimit;
    rng.seed(params.seed);

    cerr << "[MACUT] Gerando populacao inicial (p=" << params.p << ")...\n";
    vector<Solution> pop = generatePopulation(params.p, params.Piter, params.MaxIter, params.gamma, params.maxt);
    
    if (pop.empty()) {
        cerr << "[MACUT] Falha ao gerar populacao inicial (tempo esgotado?).\n";
        Solution s; 
        s.init(total_nbr_v);
        randomPartition(s, total_nbr_v);
        computeGainsFromScratch(s);
        s.fitness = computeFitnessFromScratch(s);
        return s;
    }

    Solution best = *max_element(pop.begin(), pop.end(),
                                  [](const Solution &a, const Solution &b) {
                                      return a.fitness < b.fitness;
                                  });

    cerr << "[MACUT] Melhor da populacao inicial: f = " << fixed << setprecision(3) << best.fitness << "\n";
    
    generations = 0;
    while (!timeUp()) {
        generations++;

        int maxM = min(4, (int)pop.size());
        int m = (maxM >= 2) ? randInt(2, maxM) : maxM; 
        vector<int> idx(pop.size());
        iota(idx.begin(), idx.end(), 0);
        shuffle(idx.begin(), idx.end(), rng);
        vector<const Solution*> parents;
        
        for (int i = 0; i < m; i++) parents.push_back(&pop[idx[i]]);

        Solution offspring = recombination(parents);
        Solution improved;
        perturbationBasedTabuSearch(offspring, params.Piter, params.MaxIter, params.gamma, params.maxt, improved);

        if (improved.fitness > best.fitness) {
            best = improved;
            cerr << "[MACUT] Geracao " << generations
                 << " | nova melhor solucao: f = " << best.fitness
                 << " | t = " << fixed << setprecision(1) << elapsed() << "s\n";
        }

        poolUpdating(improved, pop);
    }

    return best;
}

static void printUsage(const char *prog) {
    cerr << "Uso: " << prog << " <arquivo_grafo> [opcoes]\n"
         << "  -p <int>        tamanho da populacao          (default 10)\n"
         << "  -piter <int>    Piter                          (default 500)\n"
         << "  -gamma <int>    forca da perturbacao           (default 150)\n"
         << "  -maxiter <long> MaxIter da busca tabu           (default 1000000)\n"
         << "  -maxt <int>     maxt                             (default time(200))\n"
         << "  -time <double>  tempo limite total em segundos (default 60)\n"
         << "  -seed <int>     semente aleatoria               (default time(0))\n"
         << "  -out <file>     grava a melhor particao encontrada\n";
}

int main(int argc, char **argv) {
    ios::sync_with_stdio(false);

    if (argc < 2) { 
        printUsage(argv[0]); 
        return 1; 
    }

    string graphArg = argv[1];
    MacutParams params;

    load(graphArg);
    setup_data();

    int argi = 2;

    for (; argi < argc; argi++) {
        string a = argv[argi];
        auto next = [&]() { 
            return string(argv[++argi]); 
        };
        if      (a == "-p")       params.p         = stoi(next());
        else if (a == "-piter")   params.Piter     = stoi(next());
        else if (a == "-gamma")   params.gamma     = stoi(next());
        else if (a == "-maxiter") params.MaxIter   = stoll(next());
        else if (a == "-maxt")    params.maxt      = stoi(next());
        else if (a == "-time")    params.timeLimit = stod(next());
        else if (a == "-seed")    params.seed      = (unsigned)stoul(next());
        else if (a == "-out")     params.outFile   = next();
        else { 
            cerr << "Opcao desconhecida: " << a << "\n"; 
            printUsage(argv[0]); return 1; 
        }
    }

    cerr << "[MACUT] Grafo: n=" << total_nbr_v << ", m=" << nbr_edges << "\n";
    cerr << "[MACUT] Parametros: p=" << params.p << " Piter=" << params.Piter
         << " gamma=" << params.gamma << " MaxIter=" << params.MaxIter << " maxt=" << params.maxt
         << " time=" << params.timeLimit << "s seed=" << params.seed << "\n";

    ll generations = 0;
    Solution best  = macut(params, generations);
    float verify  = computeFitnessFromScratch(best);

    cout << "==================================================\n";
    cout << fixed << setprecision(3);
    cout << "Melhor valor de corte encontrado (f_best):  " << best.fitness << "\n";
    cout << "Verificacao independente do fitness:        " << verify << "\n";
    cout << "Numero de geracoes (ciclos evolutivos):     " << generations << "\n";
    cout << "Tempo total:                                " << fixed << setprecision(2) << elapsed() << " s\n";
    cout << "Iteracoes total:                            " << iterations_count << "\n";
    cout << "Avaliacoes total:                           " << avaliations_count << "\n";
    cout << "==================================================\n";

    if (!params.outFile.empty()) {
        ofstream fout(params.outFile);
        fout << "# Particao encontrada por MACUT. f = " << best.fitness << "\n";
        fout << "# vertice(1-indexado) subconjunto(1 ou 2)\n";
        for (int v = 0; v < total_nbr_v; v++)
            fout << (v + 1) << " " << (best.part[v] + 1) << "\n";
        cerr << "[MACUT] Particao salva em " << params.outFile << "\n";
    }

    return 0;
}