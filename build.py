import os
import sys
import time
import subprocess

def run_step(step_name: str, cmd: list[str]):
    print(f"\n=======================================================")
    print(f"BUILD STEP: {step_name}")
    print(f"=======================================================")
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.perf_counter()
    
    if res.returncode != 0:
        print(f"FAILED: {step_name}")
        print(res.stderr or res.stdout)
        sys.exit(1)
        
    print(res.stdout)
    print(f"COMPLETED in {(t1 - t0):.2f} seconds.")

def main():
    print("\n=======================================================")
    print("BUILDING SWIGGY CONSTRAINT-AWARE HYBRID FOOD SEARCH ENGINE")
    print("=======================================================\n")
    t_start = time.perf_counter()

    # Step 1: Preprocessing & Offline Embeddings
    run_step("1. Preprocessing & Offline Embedding Persistence", [sys.executable, "data/preprocess.py"])

    # Step 2: 200 Evaluation Queries Dataset Generation
    run_step("2. Generating 200 Categorized Evaluation Queries", [sys.executable, "data/generate_200_queries.py"])

    # Step 3: Realistic Query Click Log Generation
    run_step("3. Realistic Query Click Log Generation (Position Bias)", [sys.executable, "data/generate_query_click_logs.py"])

    # Step 4: Learning-to-Rank GBDT Model Training
    run_step("4. Training Learning-to-Rank GBDT Model on Click Logs", [sys.executable, "train_ltr.py"])

    # Step 5: Offline Evaluation & Profiling (200 Benchmark Queries)
    run_step("5. Offline Evaluation & Performance Profiling (200 Queries)", [sys.executable, "evaluate.py"])

    # Step 6: Feature Ablation & Comparative Benchmarks
    run_step("6. Feature Ablation Experiment", [sys.executable, "ablation.py"])
    run_step("7. BM25 vs TF-IDF Lexical Benchmark", [sys.executable, "benchmark_bm25_vs_tfidf.py"])

    # Step 7: Vector Retrieval & Embedding Model Benchmarks
    run_step("8. FAISS vs NumPy Matrix Multiplication Benchmark", [sys.executable, "benchmark_faiss_vs_numpy.py"])
    run_step("9. Comparative Retrieval & Query Cache Benchmark", [sys.executable, "benchmark_embedding_models.py"])

    # Step 8: Pytest Unit & Integration Test Suite
    run_step("10. Pytest Unit & Integration Test Suite", [sys.executable, "-m", "pytest", "-v"])

    t_end = time.perf_counter()
    print("\n=======================================================")
    print(f"FULL REPOSITORY BUILD SUCCESSFUL in {(t_end - t_start):.2f} seconds!")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
