import os
import time
import subprocess
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - SUPERVISOR - %(levelname)s - %(message)s')

class AgentSupervisor:
    def __init__(self):
        self.check_interval = 60  # Check GitHub every 60 seconds
        self.process = None
        
        # Determine the correct python executable path based on environment
        self.python_exe = "python"
        if os.path.exists("../venv/Scripts/python.exe"):
            self.python_exe = os.path.abspath("../venv/Scripts/python.exe")
            
        self.target_script = "main.py"

    def start_agent(self):
        """Starts the main.py trading agent as a subprocess."""
        if self.process is None or self.process.poll() is not None:
            logging.info(f"Starting {self.target_script}...")
            self.process = subprocess.Popen([self.python_exe, self.target_script])
            logging.info(f"Agent started with PID {self.process.pid}.")

    def stop_agent(self):
        """Gracefully stops the trading agent."""
        if self.process and self.process.poll() is None:
            logging.info(f"Stopping {self.target_script} (PID {self.process.pid})...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logging.warning("Process did not terminate cleanly. Force killing...")
                self.process.kill()
            logging.info("Agent stopped successfully.")
            self.process = None

    def check_for_updates(self) -> bool:
        """Runs 'git fetch' and 'git status' to check if the remote bridge has new code."""
        try:
            # Fetch the latest changes from the remote without merging
            subprocess.run(["git", "fetch", "origin"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Check if our local branch is behind the remote
            result = subprocess.run(["git", "status", "-uno"], capture_output=True, text=True)
            if "Your branch is behind" in result.stdout:
                return True
        except Exception as e:
            logging.error(f"Failed to check for updates: {e}")
        return False

    def pull_updates(self):
        """Pulls the new code from GitHub."""
        try:
            logging.info("New code detected on the Bridge. Pulling updates...")
            subprocess.run(["git", "pull"], check=True)
            logging.info("Updates pulled successfully.")
            return True
        except Exception as e:
            logging.error(f"Failed to pull updates: {e}")
            return False

    def run(self):
        """Main supervisor loop."""
        logging.info("Starting Supervisor Auto-Deployment Watchdog...")
        self.start_agent()

        while True:
            time.sleep(self.check_interval)
            
            # 1. Check if the agent crashed unexpectedly
            if self.process.poll() is not None:
                logging.warning("Agent process crashed! Restarting immediately...")
                self.start_agent()
                continue
                
            # 2. Check for new code on GitHub
            if self.check_for_updates():
                # Pause the agent safely
                self.stop_agent()
                
                # Pull the new brain files
                if self.pull_updates():
                    logging.info("Codebase updated. Rebooting Agent with new logic...")
                else:
                    logging.error("Update failed. Rebooting Agent with existing logic...")
                
                # Restart the agent
                self.start_agent()

if __name__ == "__main__":
    supervisor = AgentSupervisor()
    try:
        supervisor.run()
    except KeyboardInterrupt:
        logging.info("Supervisor stopped by user.")
        supervisor.stop_agent()
