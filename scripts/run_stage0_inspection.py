import os
import sys
import shutil
import psutil
import torch
import gradio as gr
from pathlib import Path

def main():
    print("=== ENVIRONMENT INSPECTION ===")
    print("Project Root:", os.getcwd())
    print("Python:", sys.version.replace('\n', ' '))
    print("PyTorch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
    print("Gradio:", gr.__version__)

    mem = psutil.virtual_memory()
    print(f"RAM Total: {mem.total / (1024**3):.2f} GB | Used: {mem.used / (1024**3):.2f} GB | Avail: {mem.available / (1024**3):.2f} GB")

    total, used, free = shutil.disk_usage('.')
    print(f"Disk Total: {total / (1024**3):.2f} GB | Used: {used / (1024**3):.2f} GB | Free: {free / (1024**3):.2f} GB")

    # Checkpoints
    ckpt = Path('checkpoints/baseline_run/best_model.pth')
    print("Baseline Best Checkpoint:", ckpt.exists(), f"({ckpt.stat().st_size} bytes)" if ckpt.exists() else "Missing")

    # Phase F.4 Reports
    f4_reports = [
        'reports/phase_f4_implementation_audit.md',
        'reports/phase_f4_visual_quality_assessment.md',
        'reports/phase_f4_failure_analysis.md',
        'reports/phase_f4_independent_evaluation_plan.md',
        'reports/phase_f4_final_recommendations.md'
    ]
    print("All F.4 reports present:", all(os.path.exists(r) for r in f4_reports))

    # Tests
    tests = ['test_pipeline.py', 'tests/test_inference_pipeline.py']
    print("Existing test suites present:", all(os.path.exists(t) for t in tests))

    # Create safe backup of source code
    backup_dir = Path('backup/pre_phase_g_src')
    backup_dir.mkdir(parents=True, exist_ok=True)
    for src_f in Path('src').glob('*.py'):
        shutil.copy2(src_f, backup_dir / src_f.name)
    for app_f in Path('app').glob('*.py'):
        shutil.copy2(app_f, backup_dir / app_f.name)
    num_backed_up = len(list(backup_dir.glob("*.py")))
    print(f"Source backup created in: {backup_dir} ({num_backed_up} files)")

if __name__ == '__main__':
    main()
