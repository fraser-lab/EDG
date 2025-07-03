"""Test the EDG configuration system.

This module tests the YAML configuration loading, validation,
parameter overrides, and CLI integration.
"""

import pytest
import tempfile
import yaml
from pathlib import Path

from edg.config import (
    load_config, 
    save_config,
    ExperimentConfig,
    merge_overrides
)
from edg.config.schedules import (
    ConstantScheduleConfig,
    PiecewiseScheduleConfig,
    ExponentialInterpolationConfig,
    parse_schedule_config
)
from edg.cli import parse_overrides, parse_override_value


class TestConfigLoading:
    """Test configuration loading and validation."""
    
    def test_load_simple_config(self):
        """Test loading a simple configuration file."""
        # Create a minimal config
        config_data = {
            "name": "test_experiment",
            "structure": {
                "structure_path": "/fake/path/structure.cif"
            },
            "density": {
                "map_path": "/fake/path/density.ccp4",
                "resolution": 2.0
            },
            "output_dir": "/fake/output",
            "input_data_dir": "/fake/input"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            # Load config without validation (since files don't exist)
            from edg.config.config_loader import create_experiment_config, parse_schedule_configs
            
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            config_data = parse_schedule_configs(config_data)
            config = create_experiment_config(config_data)
            
            assert config.name == "test_experiment"
            assert config.structure.structure_path == "/fake/path/structure.cif"
            assert config.density.map_path == "/fake/path/density.ccp4"
            assert config.density.resolution == 2.0
            assert config.output_dir == "/fake/output"
            assert config.input_data_dir == "/fake/input"
            
            # Check defaults are applied
            assert config.model.version == "boltz2"
            assert config.diffusion.num_steps == 200
            assert config.steering.enabled == True
            assert config.steering.num_particles == 3
            
        finally:
            Path(config_path).unlink()
    
    def test_config_validation_missing_files(self):
        """Test configuration validation with missing files."""
        config_data = {
            "name": "test_experiment",
            "structure": {
                "structure_path": "/nonexistent/structure.cif"
            },
            "density": {
                "map_path": "/nonexistent/density.ccp4",
                "resolution": 2.0
            },
            "output_dir": "/fake/output",
            "input_data_dir": "/fake/input"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Structure file not found"):
                load_config(config_path)
        finally:
            Path(config_path).unlink()
    
    def test_config_validation_missing_resolution(self):
        """Test validation error for missing resolution with CCP4 file."""
        config_data = {
            "name": "test_experiment",
            "structure": {
                "structure_path": "/fake/structure.cif"  # Will fail file check first
            },
            "density": {
                "map_path": "/fake/density.ccp4",
                # Missing resolution for CCP4 file
            },
            "output_dir": "/fake/output",
            "input_data_dir": "/fake/input"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            # Will fail on file not found, but let's test the structure validation directly
            from edg.config.config_schema import ExperimentConfig, StructureConfig, DensityConfig
            config = ExperimentConfig(
                name="test",
                structure=StructureConfig(structure_path="/fake/structure.cif"),
                density=DensityConfig(map_path="/fake/density.ccp4"),  # No resolution
                output_dir="/fake/output",
                input_data_dir="/fake/input"
            )
            errors = config.validate()
            assert any("Resolution must be specified" in error for error in errors)
            
        finally:
            Path(config_path).unlink()


class TestSchedules:
    """Test parameter schedule configurations."""
    
    def test_constant_schedule(self):
        """Test constant schedule configuration."""
        schedule = ConstantScheduleConfig(value=0.5)
        assert schedule.to_schedule() == 0.5
    
    def test_piecewise_schedule(self):
        """Test piecewise schedule configuration."""
        schedule = PiecewiseScheduleConfig(
            breakpoints=[0.25, 0.75],
            values=[0, 1.0, 0]
        )
        actual_schedule = schedule.to_schedule()
        assert hasattr(actual_schedule, 'compute')  # Should be a PiecewiseSchedule object
    
    def test_exponential_schedule(self):
        """Test exponential interpolation schedule."""
        schedule = ExponentialInterpolationConfig(
            start=0.1,
            end=1.0,
            alpha=2.0
        )
        actual_schedule = schedule.to_schedule()
        assert hasattr(actual_schedule, 'compute')
    
    def test_parse_schedule_from_dict(self):
        """Test parsing schedule from dictionary representation."""
        # Constant schedule
        constant_dict = {"type": "constant", "value": 0.5}
        schedule = parse_schedule_config(constant_dict)
        assert isinstance(schedule, ConstantScheduleConfig)
        assert schedule.value == 0.5
        
        # Piecewise schedule
        piecewise_dict = {
            "type": "piecewise",
            "breakpoints": [0.25, 0.75],
            "values": [0, 1.0, 0]
        }
        schedule = parse_schedule_config(piecewise_dict)
        assert isinstance(schedule, PiecewiseScheduleConfig)
        assert schedule.breakpoints == [0.25, 0.75]
        assert schedule.values == [0, 1.0, 0]  # Simple values should remain unchanged
    
    def test_parse_schedule_nested(self):
        """Test parsing nested schedule configurations."""
        nested_dict = {
            "type": "piecewise",
            "breakpoints": [0.5],
            "values": [
                {"type": "constant", "value": 0.1},
                {"type": "exponential", "start": 0.1, "end": 1.0, "alpha": 2.0}
            ]
        }
        schedule = parse_schedule_config(nested_dict)
        assert isinstance(schedule, PiecewiseScheduleConfig)
        assert isinstance(schedule.values[0], ConstantScheduleConfig)
        assert isinstance(schedule.values[1], ExponentialInterpolationConfig)


class TestOverrides:
    """Test command-line parameter overrides."""
    
    def test_merge_simple_overrides(self):
        """Test merging simple parameter overrides."""
        config_data = {
            "diffusion": {"num_steps": 200},
            "optimization": {"ensemble_size": 1}
        }
        
        overrides = {"num_steps": 300, "ensemble_size": 4}
        merged = merge_overrides(config_data, overrides)
        
        assert merged["diffusion"]["num_steps"] == 300
        assert merged["optimization"]["ensemble_size"] == 4
    
    def test_merge_nested_overrides(self):
        """Test merging nested parameter overrides with dot notation."""
        config_data = {
            "density_guidance": {"base_weight": 0.4},
            "steering": {"num_particles": 3}
        }
        
        overrides = {
            "density_guidance.base_weight": 0.8,
            "steering.num_particles": 5
        }
        merged = merge_overrides(config_data, overrides)
        
        assert merged["density_guidance"]["base_weight"] == 0.8
        assert merged["steering"]["num_particles"] == 5
    
    def test_merge_creates_new_sections(self):
        """Test that overrides can create new configuration sections."""
        config_data = {"name": "test"}
        
        overrides = {"new_section.new_param": "value"}
        merged = merge_overrides(config_data, overrides)
        
        assert merged["new_section"]["new_param"] == "value"
    
    def test_parse_override_value_types(self):
        """Test parsing override values to correct Python types."""
        assert parse_override_value("123") == 123
        assert parse_override_value("123.45") == 123.45
        assert parse_override_value("true") == True
        assert parse_override_value("false") == False
        assert parse_override_value("string_value") == "string_value"
    
    def test_parse_override_value_yaml(self):
        """Test parsing complex override values using YAML."""
        # List
        assert parse_override_value("[1, 2, 3]") == [1, 2, 3]
        
        # Dict (should fallback to string for complex cases)
        result = parse_override_value('{"key": "value"}')
        assert result == {"key": "value"}


class TestConfigSaving:
    """Test configuration saving functionality."""
    
    def test_save_and_load_config(self):
        """Test saving configuration to file and loading it back."""
        from edg.config.config_schema import StructureConfig, DensityConfig
        
        # Create a test config
        config = ExperimentConfig(
            name="test_save",
            structure=StructureConfig(structure_path="/fake/structure.cif"),
            density=DensityConfig(map_path="/fake/density.ccp4", resolution=2.0),
            output_dir="/fake/output",
            input_data_dir="/fake/input"
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            save_path = f.name
        
        try:
            # Save config
            save_config(config, save_path)
            
            # Load it back without validation (since files don't exist)
            from edg.config.config_loader import create_experiment_config, parse_schedule_configs
            
            with open(save_path, 'r') as f:
                loaded_config_data = yaml.safe_load(f)
            loaded_config_data = parse_schedule_configs(loaded_config_data)
            loaded_config = create_experiment_config(loaded_config_data)
            
            # Compare key fields
            assert loaded_config.name == config.name
            assert loaded_config.structure.structure_path == config.structure.structure_path
            assert loaded_config.density.map_path == config.density.map_path
            assert loaded_config.density.resolution == config.density.resolution
            
        finally:
            Path(save_path).unlink()


class TestCLIIntegration:
    """Test CLI argument parsing and integration."""
    
    def test_parse_override_arguments(self):
        """Test parsing command-line override arguments."""
        # Mock argparse namespace
        class MockArgs:
            def __init__(self):
                self.num_steps = 300
                self.resolution = 1.5
                self.guidance_weight = 0.8
                self.substructure_enabled = True
                self.guidance_update = True
                self.no_guidance_update = False
                self.override = ["custom.param=value", "another=123"]
                # Set all other attributes to None
                for attr in ['ensemble_size', 'step_scale', 'map_path', 'em_mode', 
                           'structure_path', 'resampling_weight', 'num_particles',
                           'learning_rate', 'solver_type', 'max_iterations',
                           'model_version', 'checkpoint_path', 'device', 'output_dir',
                           'name', 'substructure_selection']:
                    setattr(self, attr, None)
        
        args = MockArgs()
        overrides = parse_overrides(args)
        
        assert overrides["num_steps"] == 300
        assert overrides["resolution"] == 1.5
        assert overrides["guidance_weight"] == 0.8
        assert overrides["substructure_enabled"] == True
        assert overrides["guidance_update"] == True
        assert overrides["custom.param"] == "value"
        assert overrides["another"] == 123


def test_template_configs_are_valid():
    """Test that all template configurations are valid."""
    templates_dir = Path("configs/templates")
    if not templates_dir.exists():
        pytest.skip("Templates directory not found")
    
    for config_file in templates_dir.glob("*.yaml"):
        print(f"Testing template: {config_file}")
        
        # Try to load the config (will fail validation on missing files, but structure should be OK)
        try:
            with open(config_file) as f:
                config_data = yaml.safe_load(f)
            
            # Check basic structure
            assert "name" in config_data
            assert "structure" in config_data
            assert "density" in config_data
            assert "output_dir" in config_data
            assert "input_data_dir" in config_data
            
            print(f"✓ Template {config_file.name} has valid structure")
            
        except Exception as e:
            pytest.fail(f"Template {config_file.name} failed basic validation: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])