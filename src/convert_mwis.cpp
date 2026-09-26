
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <cerrno>
#include <cstdlib>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;
using u64 = std::uint64_t;

void escrever_peso(std::ostream& out, std::int64_t peso_u, std::int64_t peso_v) { 
    __int128 soma = static_cast<__int128>(peso_u) + peso_v;
    if (soma < 0) {
        out << '-';
        soma = -soma;
    }
    out << static_cast<u64>(soma / 1000) << '.';
    unsigned resto = static_cast<unsigned>(soma % 1000);
    out << static_cast<char>('0' + resto / 100)
        << static_cast<char>('0' + (resto / 10) % 10)
        << static_cast<char>('0' + resto % 10);
}

void ordenar_sem_duplicatas(const fs::path& bruto, const fs::path& ordenado, const fs::path& pasta_temp) {
    pid_t pid = fork();
    if (pid < 0) throw std::runtime_error("Falha ao iniciar o sort");
    if (pid == 0) {
        setenv("LC_ALL", "C", 1);
        execlp("sort", "sort", "-n", "-k1,1", "-k2,2", "-u",
               "-S", "512M", "-T", pasta_temp.c_str(), "-o",
               ordenado.c_str(), bruto.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }
    int status = 0;
    pid_t aguardado;
    do { 
        aguardado = waitpid(pid, &status, 0); 
    }while (aguardado == -1 && errno == EINTR);
    if (aguardado == -1 || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
        throw std::runtime_error("O comando sort falhou. Verifique espaco em disco.");
}

void converter(const fs::path& entrada, const fs::path& saida, const std::string& nome) {
    const fs::path pasta = entrada / nome;
    const fs::path arq_grafo = pasta / "conflict_graph.txt";
    const fs::path arq_pesos = pasta / "node_weights.txt";
    if (!fs::exists(arq_grafo) || !fs::exists(arq_pesos))
        throw std::runtime_error("Arquivos ausentes na instancia " + nome);

    std::ifstream grafo(arq_grafo);
    if (!grafo) throw std::runtime_error("Nao foi possivel abrir " + arq_grafo.string());
    
    u64 n = 0, m_original = 0;
    if (!(grafo >> n >> m_original) || n == 0 || n >= static_cast<u64>(std::numeric_limits<std::size_t>::max()))
        throw std::runtime_error("Cabecalho invalido em " + arq_grafo.string());

    std::vector<std::int64_t> pesos(static_cast<std::size_t>(n) + 1);
    std::vector<bool> lido(static_cast<std::size_t>(n) + 1, false);
    std::ifstream arq_p(arq_pesos);
    if (!arq_p) throw std::runtime_error("Nao foi possivel abrir " + arq_pesos.string());
    
    for (u64 i = 0; i < n; ++i) {
        u64 vertice = 0;
        std::int64_t peso = 0;
        if (!(arq_p >> vertice >> peso) || vertice == 0 || vertice > n || lido[vertice])
            throw std::runtime_error("Pesos incompletos, repetidos ou invalidos em " + nome);
        pesos[vertice] = peso;
        lido[vertice] = true;
    }

    std::string sobra;
    if (arq_p >> sobra) throw std::runtime_error("Dados extras em node_weights.txt: " + nome);

    const fs::path temp = saida / ".tmp";
    fs::create_directories(temp);
    const fs::path bruto = temp / (nome + ".raw");
    const fs::path ordenado = temp / (nome + ".sorted");
    const fs::path parcial = saida / (nome + ".txt.part");
    const fs::path final = saida / (nome + ".txt");

    try {
        {
            std::ofstream raw(bruto, std::ios::trunc);
            if (!raw) throw std::runtime_error("Falha ao criar arquivo temporario: " + nome);
            for (u64 i = 0; i < m_original; ++i) {
                u64 u = 0, v = 0;
                if (!(grafo >> u >> v) || u == 0 || v == 0 || u > n || v > n)
                    throw std::runtime_error("Aresta ausente ou invalida em " + nome);
                if (u > v) std::swap(u, v);
                raw << u << ' ' << v << '\n';
            }
            if (grafo >> sobra)
                throw std::runtime_error("Ha mais arestas do que indica o cabecalho: " + nome);
            raw.close();
            if (!raw) throw std::runtime_error("Falha ao gravar arestas de " + nome);
        }

        ordenar_sem_duplicatas(bruto, ordenado, temp);
        fs::remove(bruto);

        std::ifstream unicas(ordenado);
        if (!unicas) throw std::runtime_error("Falha ao ler arestas ordenadas: " + nome);
        u64 m_final = 0;
        std::string linha;

        while (std::getline(unicas, linha)) {
            if (!linha.empty()) ++m_final;
        }

        if (unicas.bad()) throw std::runtime_error("Falha ao contar arestas: " + nome);
        unicas.clear();
        unicas.seekg(0);

        std::ofstream out(parcial, std::ios::trunc);
        
        if (!out) throw std::runtime_error("Falha ao criar saida: " + nome);
        
        out << n << ' ' << n << ' ' << m_final << '\n';
        u64 u = 0, v = 0, escritas = 0;
        
        while (unicas >> u >> v) {
            if (u == 0 || v == 0 || u > n || v > n)
                throw std::runtime_error("Aresta ordenada invalida: " + nome);
            out << u << ' ' << v << ' ';
            escrever_peso(out, pesos[u], pesos[v]);
            out << '\n';
            ++escritas;
        }
        if (unicas.bad() || escritas != m_final)
            throw std::runtime_error("Falha ao processar arestas ordenadas: " + nome);
        
        out.close();
        
        if (!out) throw std::runtime_error("Falha ao gravar a saida: " + nome);
        
        fs::rename(parcial, final);
        fs::remove(ordenado);
        std::cout << nome << ": " << n << " vertices, " << m_final
                  << " arestas (" << (m_original - m_final)
                  << " duplicatas removidas) -> " << final << '\n';
    } catch (...) {
        std::error_code ec;
        fs::remove(bruto, ec);
        fs::remove(ordenado, ec);
        fs::remove(parcial, ec);
        throw;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3 && argc != 4) {
        std::cerr << "Uso: " << argv[0] << " <instancias_mwis> <graph_instances> [NOME_DA_INSTANCIA]\n";
        return 1;
    }
    try {
        const fs::path entrada = argv[1], saida = argv[2];
        if (!fs::is_directory(entrada))
            throw std::runtime_error("Pasta de instancias nao encontrada: " + entrada.string());
        fs::create_directories(saida);
        if (argc == 4) {
            converter(entrada, saida, argv[3]);
        } else {
            std::vector<std::string> nomes;
            for (const auto& item : fs::directory_iterator(entrada))
                if (item.is_directory() && fs::exists(item.path() / "conflict_graph.txt"))
                    nomes.push_back(item.path().filename().string());
            std::sort(nomes.begin(), nomes.end());
            if (nomes.empty())
                throw std::runtime_error("Nenhuma instancia encontrada em " + entrada.string());
            for (const auto& nome : nomes) converter(entrada, saida, nome);
        }
    } catch (const std::exception& e) {
        std::cerr << "Erro: " << e.what() << '\n';
        return 1;
    }
}
