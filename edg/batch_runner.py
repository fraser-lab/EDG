"""Batch experiment runner for EDG density-guided diffusion experiments.

This module provides functionality to run multiple experiments in batch:
- Queue management for sequential processing
- Progress tracking and error handling per experiment
- Reuses existing run_experiment() function
- Supports continue-on-error mode
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import traceback
import json

from edg.config import BatchExperimentConfig, ExperimentConfig
from edg.experiment_runner import run_experiment

logger = logging.getLogger(__name__)


class BatchRunner:
    """Handles batch execution of EDG experiments."""
    
    def __init__(self, batch_config: BatchExperimentConfig):
        """Initialize batch runner.
        
        Parameters
        ----------
        batch_config : BatchExperimentConfig
            Batch experiment configuration
        """
        self.batch_config = batch_config
        self.results: List[Dict[str, Any]] = []
        self.failed_experiments: List[str] = []
        self.start_time: Optional[float] = None
        
    def run_batch(self) -> Dict[str, Any]:
        """Run all experiments in the batch.
        
        Returns
        -------
        Dict[str, Any]
            Batch execution results and summary
        """
        logger.info(f"Starting batch: {self.batch_config.name}")
        self.start_time = time.time()
        
        # Get all experiment configurations
        experiment_configs = self.batch_config.get_experiment_configs()
        total_experiments = len(experiment_configs)
        
        if total_experiments == 0:
            logger.warning("No experiments found in batch configuration")
            return self._create_summary()
        
        logger.info(f"Found {total_experiments} experiments to run")
        
        # Create batch output directory
        batch_output_dir = Path(self.batch_config.output_base_dir)
        batch_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save batch configuration
        batch_config_path = batch_output_dir / "batch_config.yaml"
        from edg.config import save_batch_config
        save_batch_config(self.batch_config, batch_config_path)
        logger.info(f"Saved batch configuration to {batch_config_path}")
        
        # Initialize progress tracking
        progress_file = batch_output_dir / "progress.json"
        self._save_progress(progress_file, 0, total_experiments)
        
        # Run experiments
        for i, experiment_config in enumerate(experiment_configs):
            try:
                logger.info(f"Running experiment {i+1}/{total_experiments}: {experiment_config.name}")
                
                # Run single experiment
                experiment_result = run_experiment(experiment_config)
                
                # Store result
                self.results.append({
                    "experiment_name": experiment_config.name,
                    "status": "success",
                    "result": experiment_result
                })
                
                logger.info(f"Completed experiment {i+1}/{total_experiments}: {experiment_config.name}")
                
            except Exception as e:
                error_msg = f"Experiment {experiment_config.name} failed: {str(e)}"
                logger.error(error_msg)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Full traceback:\n{traceback.format_exc()}")
                
                # Store failure
                self.results.append({
                    "experiment_name": experiment_config.name,
                    "status": "failed",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                })
                self.failed_experiments.append(experiment_config.name)
                
                # Check if we should continue
                if not self.batch_config.continue_on_error:
                    logger.error("Stopping batch execution due to error (continue_on_error=False)")
                    break
                else:
                    logger.info("Continuing with next experiment (continue_on_error=True)")
            
            # Update progress
            self._save_progress(progress_file, i+1, total_experiments)
        
        # Create final summary
        summary = self._create_summary()
        
        # Save summary
        summary_path = batch_output_dir / "batch_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Batch completed: {summary['completed_experiments']}/{summary['total_experiments']} successful")
        if summary['failed_experiments']:
            logger.warning(f"Failed experiments: {', '.join(summary['failed_experiments'])}")
        
        return summary
    
    def _save_progress(self, progress_file: Path, completed: int, total: int):
        """Save progress to file.
        
        Parameters
        ----------
        progress_file : Path
            Path to progress file
        completed : int
            Number of completed experiments
        total : int
            Total number of experiments
        """
        progress_data = {
            "batch_name": self.batch_config.name,
            "completed": completed,
            "total": total,
            "percentage": (completed / total * 100) if total > 0 else 0,
            "timestamp": time.time(),
            "elapsed_time": time.time() - self.start_time if self.start_time else 0
        }
        
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def _create_summary(self) -> Dict[str, Any]:
        """Create batch execution summary.
        
        Returns
        -------
        Dict[str, Any]
            Summary of batch execution
        """
        total_experiments = len(self.results)
        successful_experiments = sum(1 for r in self.results if r["status"] == "success")
        failed_experiments = sum(1 for r in self.results if r["status"] == "failed")
        
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        summary = {
            "batch_name": self.batch_config.name,
            "total_experiments": total_experiments,
            "completed_experiments": successful_experiments,
            "failed_experiments": failed_experiments,
            "success_rate": (successful_experiments / total_experiments * 100) if total_experiments > 0 else 0,
            "failed_experiment_names": self.failed_experiments,
            "elapsed_time_seconds": elapsed_time,
            "results": self.results
        }
        
        return summary


def run_batch_experiment(batch_config: BatchExperimentConfig) -> Dict[str, Any]:
    """Run a batch of EDG experiments.
    
    Parameters
    ----------
    batch_config : BatchExperimentConfig
        Batch experiment configuration
    
    Returns
    -------
    Dict[str, Any]
        Batch execution results and summary
    """
    runner = BatchRunner(batch_config)
    return runner.run_batch()