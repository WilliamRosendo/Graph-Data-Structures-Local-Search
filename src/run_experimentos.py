
#!/usr/bin/env python3
"""Compila e executa independentemente os algoritmos MDP e Max Bisection."""

import argparse
import subprocess
import sys
from pathlib import Path

ALGOS = {
    "csr": ("CSR", "csr"),
    "hash": ("Hash", "hash"),
    "wswap": ("WSwap", "wswap"),
    "csr-bitset": ("CSR-Bitset", "csr-bitset"),
    "csr-bb": ("CSR-BuscaBinaria", "csr-bb"),
    "csr-bitset-bb": ("CSR-Bitset-BuscaBinaria", "csr-bitset-bb"),
}

PROBLEMAS = {
    "mdp": ("mdp", "GPU-MDP", "mdp"),
    "maxbs": ("maxbisection", "GPU-MAXBS", "maxbs"),
}

MARCADOR_OK = b"\n[EXPERIMENTOS] EXIT_CODE=0\n"


def concluido(log: Path) -> bool:
    if not log.is_file():
        return False

    with log.open("rb") as arquivo:
        arquivo.seek(max(0, log.stat().st_size - 256))
        return arquivo.read().endswith(MARCADOR_OK)


def quantidade_vertices(grafo: Path) -> int:
    with grafo.open("r", encoding="utf-8") as arquivo:
        cabecalho = arquivo.readline().split()

    if len(cabecalho) != 3:
        raise ValueError(f"Cabecalho invalido em {grafo}")

    n, segunda_dimensao, _ = map(int, cabecalho)

    if n < 1 or segunda_dimensao != n:
        raise ValueError(f"Cabecalho invalido em {grafo}")

    return n


def compilar(fonte: Path, executavel: Path, log: Path, forcar: bool) -> bool:
    if not fonte.is_file():
        print(f"[ERRO] Codigo-fonte nao encontrado: {fonte}", flush=True)
        return False

    if (not forcar and executavel.is_file()
            and executavel.stat().st_mtime >= fonte.stat().st_mtime):
        print(f"[COMPILACAO] Reutilizando {executavel.name}", flush=True)
        return True

    executavel.parent.mkdir(parents=True, exist_ok=True)

    comando = [
        "nvcc", str(fonte), "-o", str(executavel),
        "-O3", "-ccbin", "g++-10"
    ]

    print(f"[COMPILACAO] {' '.join(comando)}", flush=True)

    with log.open("w", encoding="utf-8") as saida:
        saida.write("Comando: " + " ".join(comando) + "\n\n")
        saida.flush()

        try:
            resultado = subprocess.run(
                comando,
                cwd=fonte.parent,
                stdout=saida,
                stderr=subprocess.STDOUT,
                check=False
            )
            codigo = resultado.returncode

        except OSError as erro:
            saida.write(f"\nFalha ao compilar: {erro}\n")
            codigo = 127

        saida.write(f"\n[EXPERIMENTOS] EXIT_CODE={codigo}\n")

    if codigo != 0:
        print(f"[ERRO] Falha na compilacao. Consulte {log}", flush=True)
        return False

    return True


def executar(comando: list[str], log: Path, cwd: Path) -> bool:
    # Cada chamada inicia um novo processo independente.
    with log.open("w", encoding="utf-8") as saida:
        saida.write("Comando: " + " ".join(comando) + "\n\n")
        saida.flush()

        try:
            resultado = subprocess.run(
                comando,
                cwd=cwd,
                stdout=saida,
                stderr=subprocess.STDOUT,
                check=False
            )
            codigo = resultado.returncode

        except OSError as erro:
            saida.write(f"\nFalha ao executar: {erro}\n")
            codigo = 127

        saida.write(f"\n[EXPERIMENTOS] EXIT_CODE={codigo}\n")

    return codigo == 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Executa MDP e Max Bisection nas instancias convertidas."
    )

    parser.add_argument(
        "--problema",
        choices=["todos", *PROBLEMAS],
        default="todos"
    )
    parser.add_argument(
        "--alg",
        choices=["todos", *ALGOS],
        default="todos"
    )
    parser.add_argument("--instancia", help="Ex.: MR-D-01; padrao: todas")
    parser.add_argument("--repeticoes", type=int, default=10)
    parser.add_argument("--tempo", type=int, default=60)
    parser.add_argument("--maxiter", type=int, default=1000000)
    parser.add_argument("--maxt", type=int, default=200)
    parser.add_argument(
        "--forcar",
        action="store_true",
        help="Refaz as execucoes concluidas e recompila"
    )

    args = parser.parse_args()

    if min(args.repeticoes, args.tempo, args.maxiter, args.maxt) < 1:
        parser.error("Os parametros numericos devem ser positivos")

    pasta_src = Path(__file__).resolve().parent
    raiz = pasta_src.parent
    pasta_grafos = raiz / "graph_instances"

    if args.instancia:
        nome = Path(args.instancia).name.removesuffix(".txt")
        grafos = [pasta_grafos / (nome + ".txt")]
    else:
        grafos = sorted(pasta_grafos.glob("*.txt"))

    if not grafos or any(not g.is_file() for g in grafos):
        parser.error(f"Instancias nao encontradas em {pasta_grafos}")

    problemas = (
        list(PROBLEMAS) if args.problema == "todos"
        else [args.problema]
    )

    algoritmos = (
        list(ALGOS) if args.alg == "todos"
        else [args.alg]
    )

    status = {
        "ok": 0,
        "erro": 0,
        "retomadas": 0,
        "incompativeis": 0,
        "compilacao": 0
    }

    total = len(grafos) * len(problemas) * len(algoritmos) * args.repeticoes

    print(
        f"Instancias: {len(grafos)} | "
        f"Problemas: {len(problemas)} | "
        f"Algoritmos: {len(algoritmos)} | "
        f"Repeticoes: {args.repeticoes}"
    )
    print(f"Execucoes planejadas: {total}", flush=True)

    for problema in problemas:
        subpasta, prefixo, bin_prefixo = PROBLEMAS[problema]
        pasta_codigo = pasta_src / subpasta

        for algoritmo in algoritmos:
            sufixo, nome_bin = ALGOS[algoritmo]

            pasta_resultados = pasta_codigo / "Resultados" / sufixo
            pasta_resultados.mkdir(parents=True, exist_ok=True)

            fonte = pasta_codigo / (prefixo + "-" + sufixo + ".cu")
            executavel = raiz / "bin" / (bin_prefixo + "_" + nome_bin)

            if not compilar(
                fonte, executavel,
                pasta_resultados / "compilacao.txt",
                args.forcar
            ):
                status["compilacao"] += 1
                continue

            for grafo in grafos:
                try:
                    n = quantidade_vertices(grafo)

                except (OSError, ValueError) as erro:
                    print(f"[ERRO] {erro}", flush=True)
                    status["erro"] += args.repeticoes
                    continue

                # Protege o Max Bisection contra overflow de int32.
                movimentos = (n // 2) * (n - n // 2)
                incompativel = (
                    problema == "maxbs" and movimentos > 2**31 - 1
                )

                if incompativel:
                    print(
                        f"[INCOMPATIVEL] {grafo.stem} com {sufixo}: "
                        f"{movimentos} swaps excedem int32",
                        flush=True
                    )

                for rodada in range(1, args.repeticoes + 1):
                    log = pasta_resultados / f"{grafo.stem}_{rodada}.txt"

                    # Nao repete uma execucao que ja terminou corretamente.
                    if not args.forcar and concluido(log):
                        status["retomadas"] += 1
                        continue

                    if incompativel:
                        log.write_text(
                            f"NAO EXECUTADA: {movimentos} swaps excedem int32; "
                            "o codigo Max Bisection usa int para nbr_movs.\n",
                            encoding="utf-8"
                        )
                        status["incompativeis"] += 1
                        continue

                    if problema == "mdp":
                        comando = [
                            str(executavel), str(grafo),
                            "sparse", str(args.tempo), "1"
                        ]
                    else:
                        comando = [
                            str(executavel), str(grafo),
                            "-time", str(args.tempo),
                            "-maxiter", str(args.maxiter),
                            "-maxt", str(args.maxt)
                        ]

                    print(
                        f"[{problema.upper()} / {sufixo}] "
                        f"{grafo.stem} - rodada "
                        f"{rodada}/{args.repeticoes}",
                        flush=True
                    )

                    if executar(comando, log, pasta_codigo):
                        status["ok"] += 1
                    else:
                        status["erro"] += 1
                        print(
                            f"[ERRO] Execucao falhou. Verifique {log}",
                            flush=True
                        )

    print("\nResumo:", flush=True)
    for nome, quantidade in status.items():
        print(f"  {nome}: {quantidade}", flush=True)

    if status["erro"] or status["compilacao"] or status["incompativeis"]:
        return 1

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(
            "\nInterrompido. Execute o mesmo comando para retomar.",
            file=sys.stderr
        )
        sys.exit(130)
