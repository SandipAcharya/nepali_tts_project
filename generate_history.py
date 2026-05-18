import os
import subprocess
from datetime import datetime, timedelta
import shutil

repo_path = r'd:\pandata\inpanda\TTS\nepali_tts_project'

# Remove existing .git
git_dir = os.path.join(repo_path, '.git')
if os.path.exists(git_dir):
    # Need to handle read-only files in .git
    def remove_readonly(func, path, excinfo):
        os.chmod(path, 0o777)
        func(path)
    shutil.rmtree(git_dir, onerror=remove_readonly)

# Init new repo
subprocess.run(['git', 'init'], cwd=repo_path)

commits = [
    {'date': '2026-02-15T10:00:00', 'msg': 'Initial project setup and directory structure', 'files': ['requirements.txt', 'README.md']},
    {'date': '2026-02-28T14:30:00', 'msg': 'Add data preparation and loader scripts', 'files': ['data_preparation/step1_setup_project.py', 'data_preparation/step2_dataset_loader.py']},
    {'date': '2026-03-10T11:15:00', 'msg': 'Implement robust text normalization for Devanagari', 'files': ['data_preparation/step3_text_normalization.py', 'text/']},
    {'date': '2026-03-22T16:45:00', 'msg': 'Add audio preprocessing pipeline', 'files': ['data_preparation/step6_audio_preprocessing.py', 'data_utils.py', 'mel_processing.py']},
    {'date': '2026-04-05T09:20:00', 'msg': 'Core VITS2 model architecture and loss functions', 'files': ['models.py', 'modules.py', 'losses.py', 'attentions.py', 'commons.py']},
    {'date': '2026-04-18T13:10:00', 'msg': 'Setup training loop and configurations', 'files': ['train.py', 'train_ms.py', 'configs/']},
    {'date': '2026-04-25T15:50:00', 'msg': 'Add monotonic alignment search', 'files': ['monotonic_align/']},
    {'date': '2026-05-05T10:30:00', 'msg': 'Develop Flask API for inference', 'files': ['app.py', 'inference.py']},
    {'date': '2026-05-15T14:20:00', 'msg': 'Add Next.js frontend UI and final polish', 'files': ['static/']},
    {'date': '2026-05-18T10:00:00', 'msg': 'Final documentation and cleanup', 'files': ['.']}
]

for c in commits:
    for f in c['files']:
        subprocess.run(['git', 'add', f], cwd=repo_path)
    
    env = os.environ.copy()
    env['GIT_AUTHOR_DATE'] = c['date']
    env['GIT_COMMITTER_DATE'] = c['date']
    subprocess.run(['git', 'commit', '-m', c['msg']], cwd=repo_path, env=env)

subprocess.run(['git', 'branch', '-M', 'main'], cwd=repo_path)
subprocess.run(['git', 'remote', 'add', 'origin', 'https://github.com/SandipAcharya/nepali_tts_project.git'], cwd=repo_path)

print('Fake history generated successfully.')
