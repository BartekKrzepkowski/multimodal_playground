from datetime import datetime
from pathlib import Path
from typing import Optional


class PathsManager:
    def __init__(self, custom_root: Optional[Path] = None, experiment_name: Optional[str] = None):
        """
        Initialize paths manager with optional custom root directory.
        
        Args:
            custom_root: Optional custom root directory path. If None, uses default "data" directory.
        """
        self.root_path = custom_root if custom_root else Path("data")
        if isinstance(self.root_path, str):
            self.root_path = Path(self.root_path)
        self._validate_root_directory()
        
        # Main directories
        date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.root_logs = self.root_path / experiment_name / date
        self.root_logs_checkpoints = self.root_logs / "checkpoints"
        self.root_checkpoints = Path("checkpoints")
        self.root_data = Path("data")
    

    def _validate_root_directory(self) -> None:
        """Check if root directory exists or can be created."""
        try:
            self.root_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize root directory at {self.root_path}: {e}")
    

    def create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for directory in [self.root_logs, self.root_logs_checkpoints, self.root_checkpoints, self.root_data]:
            directory.mkdir(parents=True, exist_ok=True)
