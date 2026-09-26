
# ==========================================================
# Projeto William-mestrado
# Instâncias MWIS - AWS Open Data
# ==========================================================

INST_DIR := $(CURDIR)/instancias_mwis
BUCKET := s3://mwis-vr-instances

# Instância padrão (pode ser alterada na linha de comando)
INSTANCE ?= MR-D-01

.DEFAULT_GOAL := help
.PHONY: help list download download-all extract extract-all show

# Exibir os comandos disponíveis
help:
	@echo "Comandos disponíveis:"
	@echo "  make list                        - Listar instâncias na AWS"
	@echo "  make download INSTANCE=MR-D-01   - Baixar uma instância"
	@echo "  make download-all                - Baixar todas as instâncias"
	@echo "  make extract INSTANCE=MR-D-01    - Extrair uma instância"
	@echo "  make extract-all                 - Extrair todas as instâncias"
	@echo "  make show INSTANCE=MR-D-01       - Inspecionar uma instância"
	@echo "  make convert INSTANCE=MR-D-01    - Converter uma instância"
	@echo "  make convert-all                 - Converter todas as instâncias"
	@echo "  make convert-pendentes           - Converter apenas instancias pendentes"
	@echo ""
	@echo "Algoritmos GPU MDP:"
	@echo "  make compile-mdp ALG=wswap          - Compilar algoritmo"
	@echo "  make run-mdp ALG=wswap INSTANCE=MR-D-01 TIME=60 RUNS=1"
	@echo "  ALG: wswap, csr, hash, csr-bitset, csr-bb, csr-bitset-bb"
	@echo ""
	@echo "Algoritmos GPU MAX BISECTION:"
	@echo "  make compile-maxbs ALG=wswap       - Compilar algoritmo"
	@echo "  make run-maxbs ALG=wswap INSTANCE=MR-D-01 TIME=60 MAXITER=1000000 MAXT=200"
	@echo "  ALG: wswap, csr, hash, csr-bitset, csr-bb, csr-bitset-bb"
	@echo ""
	@echo "Experimentos:"
	@echo "  make experimentos                - Executar todos os experimentos"
	@echo "  make experimentos ARGS='...'     - Executar com parametros personalizados"
	@echo ""
	@echo "Pipeline completo:"
	@echo "  make preparar  - Verificar AWS CLI e CUDA"
	@echo "  make tudo      - Executar todas as etapas"

# Listar instâncias disponíveis na AWS
list:
	aws s3 ls $(BUCKET)/ --no-sign-request

# Baixar uma instância
download:
	@mkdir -p "$(INST_DIR)"
	aws s3 cp $(BUCKET)/$(INSTANCE).tar.gz "$(INST_DIR)/" --no-sign-request

# Baixar todas as instâncias
download-all:
	@mkdir -p "$(INST_DIR)"
	aws s3 sync $(BUCKET)/ "$(INST_DIR)/" \
		--exclude "*" \
		--include "*.tar.gz" \
		--no-sign-request

# Extrair uma instância
extract:
	tar -xzf "$(INST_DIR)/$(INSTANCE).tar.gz" -C "$(INST_DIR)"

# Extrair todas as instâncias ainda não extraídas
extract-all:
	@for arquivo in "$(INST_DIR)"/*.tar.gz; do \
		[ -f "$$arquivo" ] || continue; \
		nome=$$(basename "$$arquivo" .tar.gz); \
		if [ ! -d "$(INST_DIR)/$$nome" ]; then \
			echo "Extraindo $$nome..."; \
			tar -xzf "$$arquivo" -C "$(INST_DIR)" || exit 1; \
		fi; \
	done

# Inspecionar uma instância
show:
	@echo "Instância: $(INSTANCE)"
	@head -n 1 "$(INST_DIR)/$(INSTANCE)/conflict_graph.txt"
	@echo "Primeiras 5 arestas:"
	@head -n 6 "$(INST_DIR)/$(INSTANCE)/conflict_graph.txt" | tail -n 5
	@echo "Primeiros 5 pesos:"
	@head -n 5 "$(INST_DIR)/$(INSTANCE)/node_weights.txt"



# ==========================================================
# Conversor MWIS
# ==========================================================

CXX := g++
CXXFLAGS := -O3 -std=c++17 -Wall -Wextra

CONVERTER := $(CURDIR)/bin/convert_mwis

.PHONY: convert convert-all

$(CONVERTER): src/convert_mwis.cpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $< -o $@

# Converter uma única instância
convert: $(CONVERTER)
	$(CONVERTER) "$(INST_DIR)" "$(CURDIR)/graph_instances" "$(INSTANCE)"

# Converter todas as instâncias
convert-all: $(CONVERTER)
	$(CONVERTER) "$(INST_DIR)" "$(CURDIR)/graph_instances"




# Converter apenas as instancias pendentes
.PHONY: convert-pendentes

convert-pendentes: $(CONVERTER)
	@mkdir -p "$(CURDIR)/graph_instances"
	@set -eu; \
	for pasta in "$(INST_DIR)"/*/; do \
		[ -f "$$pasta/conflict_graph.txt" ] || continue; \
		nome=$$(basename "$$pasta"); \
		saida="$(CURDIR)/graph_instances/$$nome.txt"; \
		if [ ! -s "$$saida" ]; then \
			echo "Convertendo $$nome..."; \
			"$(CONVERTER)" "$(INST_DIR)" \
				"$(CURDIR)/graph_instances" "$$nome"; \
		else \
			echo "$$nome ja convertido."; \
		fi; \
	done



# ==========================================================
# GPU MDP - Algoritmos
# ==========================================================

ALG ?= wswap
TIME ?= 60
RUNS ?= 1
DATASET ?= sparse

MDP_DIR := $(CURDIR)/src/mdp
MDP_BIN := $(CURDIR)/bin

# Arquivos de cada algoritmo
MDP_SRC_wswap := GPU-MDP-WSwap.cu
MDP_SRC_csr := GPU-MDP-CSR.cu
MDP_SRC_hash := GPU-MDP-Hash.cu
MDP_SRC_csr-bitset := GPU-MDP-CSR-Bitset.cu
MDP_SRC_csr-bb := GPU-MDP-CSR-BuscaBinaria.cu
MDP_SRC_csr-bitset-bb := GPU-MDP-CSR-Bitset-BuscaBinaria.cu

MDP_SOURCE = $(MDP_DIR)/$(MDP_SRC_$(ALG))
MDP_EXEC = $(MDP_BIN)/mdp_$(ALG)

.PHONY: compile-mdp run-mdp

# Compilar o algoritmo selecionado
compile-mdp:
	@test -f "$(MDP_SOURCE)" || { echo "Algoritmo invalido: $(ALG)"; exit 1; }
	@mkdir -p "$(MDP_BIN)"
	nvcc "$(MDP_SOURCE)" -o "$(MDP_EXEC)" -O3 -ccbin g++-10

# Executar o algoritmo selecionado
run-mdp:
	@test -x "$(MDP_EXEC)" || { echo "Execute make compile-mdp ALG=$(ALG) primeiro"; exit 1; }
	@test -f "$(CURDIR)/graph_instances/$(INSTANCE).txt" || { echo "Instancia nao encontrada: $(INSTANCE)"; exit 1; }
	"$(MDP_EXEC)" "$(CURDIR)/graph_instances/$(INSTANCE).txt" $(DATASET) $(TIME) $(RUNS)



# ==========================================================
# GPU MAX BISECTION - Algoritmos
# ==========================================================

MAXITER ?= 1000000
MAXT ?= 200

MAXBS_DIR := $(CURDIR)/src/maxbisection
MAXBS_BIN := $(CURDIR)/bin

# Arquivos de cada algoritmo
MAXBS_SRC_wswap := GPU-MAXBS-WSwap.cu
MAXBS_SRC_csr := GPU-MAXBS-CSR.cu
MAXBS_SRC_hash := GPU-MAXBS-Hash.cu
MAXBS_SRC_csr-bitset := GPU-MAXBS-CSR-Bitset.cu
MAXBS_SRC_csr-bb := GPU-MAXBS-CSR-BuscaBinaria.cu
MAXBS_SRC_csr-bitset-bb := GPU-MAXBS-CSR-Bitset-BuscaBinaria.cu

MAXBS_SOURCE = $(MAXBS_DIR)/$(MAXBS_SRC_$(ALG))
MAXBS_EXEC = $(MAXBS_BIN)/maxbs_$(ALG)

.PHONY: compile-maxbs run-maxbs

# Compilar o algoritmo selecionado
compile-maxbs:
	@test -f "$(MAXBS_SOURCE)" || { echo "Algoritmo invalido: $(ALG)"; exit 1; }
	@mkdir -p "$(MAXBS_BIN)"
	nvcc "$(MAXBS_SOURCE)" -o "$(MAXBS_EXEC)" -O3 -ccbin g++-10

# Executar o algoritmo selecionado
run-maxbs:
	@test -x "$(MAXBS_EXEC)" || { echo "Execute make compile-maxbs ALG=$(ALG) primeiro"; exit 1; }
	@test -f "$(CURDIR)/graph_instances/$(INSTANCE).txt" || { echo "Instancia nao encontrada: $(INSTANCE)"; exit 1; }
	"$(MAXBS_EXEC)" "$(CURDIR)/graph_instances/$(INSTANCE).txt" -time $(TIME) -maxiter $(MAXITER) -maxt $(MAXT)


.PHONY: experimentos

experimentos:
	python3 src/run_experimentos.py $(ARGS)




# ==========================================================
# PREPARACAO DO AMBIENTE
# ==========================================================

.PHONY: preparar

preparar:
	@set -eu; \
	echo "Verificando AWS CLI..."; \
	if ! command -v aws >/dev/null 2>&1; then \
		echo "AWS CLI nao encontrado. Instalando..."; \
		if ! command -v curl >/dev/null 2>&1 || \
		   ! command -v unzip >/dev/null 2>&1; then \
			echo "ERRO: curl e unzip sao necessarios para instalar o AWS CLI."; \
			exit 1; \
		fi; \
		TEMP=$$(mktemp -d); \
		trap 'rm -rf "$$TEMP"' EXIT; \
		curl -fsSL \
			"https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" \
			-o "$$TEMP/awscliv2.zip"; \
		unzip -q "$$TEMP/awscliv2.zip" -d "$$TEMP"; \
		sudo "$$TEMP/aws/install"; \
	else \
		echo "AWS CLI ja instalado."; \
	fi; \
	aws --version; \
	echo ""; \
	echo "Verificando CUDA..."; \
	if ! command -v nvcc >/dev/null 2>&1; then \
		echo "ERRO: CUDA (nvcc) nao encontrado."; \
		echo "Nenhuma instalacao sera realizada."; \
		exit 1; \
	fi; \
	nvcc --version; \
	echo ""; \
	echo "Ambiente verificado."





# ==========================================================
# PIPELINE COMPLETO
# ==========================================================

.PHONY: tudo

tudo:
	@echo "ETAPA 1: Verificando ambiente..."
	@$(MAKE) preparar

	@echo "ETAPA 2: Baixando instancias..."
	@$(MAKE) download-all

	@echo "ETAPA 3: Descompactando instancias..."
	@$(MAKE) extract-all

	@echo "ETAPA 4: Convertendo instancias..."
	@$(MAKE) convert-pendentes

	@echo "ETAPA 5: Compilando e executando os experimentos..."
	@$(MAKE) experimentos ARGS="$(ARGS)"

	@echo "Pipeline finalizado."
