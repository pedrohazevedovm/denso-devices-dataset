import logging
from rich.console import Console
from rich.logging import RichHandler

console = Console()

def setup_logger(name: str = "visual_servoing", level: int = logging.INFO) -> logging.Logger:
    """Configura e retorna um logger Rich formatado."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            show_path=False,
            show_time=True
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        
    return logger

logger = setup_logger()
