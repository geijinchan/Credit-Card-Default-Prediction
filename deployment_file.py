import subprocess

def run_main():
    # Calling main.py using subprocess
    subprocess.run(['python', 'main.py'], check=True)

def run_app():
    # Calling app.py using subprocess
    subprocess.run(['python', 'application.py'], check=True)

if __name__ == '__main__':
    run_main()
    run_app()
