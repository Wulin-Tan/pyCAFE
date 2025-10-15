"""Basic tests for pyCAFE functionality"""

import unittest
import os
import tempfile
import shutil
import numpy as np
from pathlib import Path

# Import pyCAFE modules
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pyCAFE
from pyCAFE.utils import get_video_info, format_time, validate_time_range


class TestUtils(unittest.TestCase):
    """Test utility functions"""
    
    def test_format_time(self):
        """Test time formatting"""
        self.assertEqual(format_time(30), "30.00s")
        self.assertEqual(format_time(90), "1m 30.0s")
        self.assertEqual(format_time(3660), "1h 1m 0s")
    
    def test_validate_time_range_valid(self):
        """Test valid time ranges"""
        try:
            validate_time_range(0.0, 1.0)
            validate_time_range(0.2, 0.8)
        except ValueError:
            self.fail("validate_time_range raised ValueError unexpectedly")
    
    def test_validate_time_range_invalid(self):
        """Test invalid time ranges"""
        with self.assertRaises(ValueError):
            validate_time_range(-0.1, 1.0)
        
        with self.assertRaises(ValueError):
            validate_time_range(0.0, 1.1)
        
        with self.assertRaises(ValueError):
            validate_time_range(0.8, 0.2)


class TestClustering(unittest.TestCase):
    """Test clustering functions"""
    
    def test_perform_kmeans_fewer_frames_than_clusters(self):
        """Test behavior when requesting more clusters than frames"""
        # Create small dataset
        frames = np.random.rand(5, 100).astype(np.float32)
        
        # Request more clusters than available frames
        selected = pyCAFE.perform_kmeans_gpu(frames, n_clusters=10)
        
        # Should return all frame indices
        self.assertEqual(len(selected), 5)
        self.assertEqual(selected, list(range(5)))
    
    def test_perform_kmeans_basic(self):
        """Test basic K-means clustering"""
        # Create dataset with 100 frames
        frames = np.random.rand(100, 50).astype(np.float32)
        
        # Request 10 clusters
        selected = pyCAFE.perform_kmeans_gpu(frames, n_clusters=10)
        
        # Should return 10 frames
        self.assertEqual(len(selected), 10)
        
        # Indices should be sorted
        self.assertEqual(selected, sorted(selected))
        
        # All indices should be unique
        self.assertEqual(len(selected), len(set(selected)))
        
        # All indices should be valid
        self.assertTrue(all(0 <= idx < 100 for idx in selected))


class TestPackageImport(unittest.TestCase):
    """Test package imports and initialization"""
    
    def test_package_version(self):
        """Test that package has version"""
        self.assertTrue(hasattr(pyCAFE, '__version__'))
        self.assertIsInstance(pyCAFE.__version__, str)
    
    def test_cuda_available_attribute(self):
        """Test CUDA_AVAILABLE attribute exists"""
        self.assertTrue(hasattr(pyCAFE, 'CUDA_AVAILABLE'))
        self.assertIsInstance(pyCAFE.CUDA_AVAILABLE, bool)
    
    def test_main_functions_available(self):
        """Test that main functions are importable"""
        self.assertTrue(hasattr(pyCAFE, 'extract_frames_kmeans_gpu'))
        self.assertTrue(hasattr(pyCAFE, 'extract_frames_kmeans_cpu'))
        self.assertTrue(hasattr(pyCAFE, 'get_video_info'))
        self.assertTrue(hasattr(pyCAFE, 'benchmark_cpu_vs_gpu'))


class TestExtraction(unittest.TestCase):
    """Test frame extraction (requires test video)"""
    
    @classmethod
    def setUpClass(cls):
        """Setup test fixtures"""
        cls.test_video = os.environ.get('TEST_VIDEO_PATH', None)
        cls.temp_dir = tempfile.mkdtemp()
    
    @classmethod
    def tearDownClass(cls):
        """Cleanup test fixtures"""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)
    
    def test_video_info_missing_file(self):
        """Test get_video_info with missing file"""
        with self.assertRaises(ValueError):
            get_video_info("nonexistent_video.mp4")
    
    @unittest.skipIf(not os.environ.get('TEST_VIDEO_PATH'), 
                     "TEST_VIDEO_PATH not set - skipping video tests")
    def test_get_video_info(self):
        """Test getting video information"""
        info = get_video_info(self.test_video)
        
        # Check required keys exist
        self.assertIn('nframes', info)
        self.assertIn('fps', info)
        self.assertIn('width', info)
        self.assertIn('height', info)
        self.assertIn('duration', info)
        
        # Check values are reasonable
        self.assertGreater(info['nframes'], 0)
        self.assertGreater(info['fps'], 0)
        self.assertGreater(info['width'], 0)
        self.assertGreater(info['height'], 0)
    
    @unittest.skipIf(not os.environ.get('TEST_VIDEO_PATH'), 
                     "TEST_VIDEO_PATH not set - skipping video tests")
    def test_extract_frames_kmeans_gpu_basic(self):
        """Test basic frame extraction"""
        output_dir = os.path.join(self.temp_dir, 'gpu_output')
        
        frames, timing = pyCAFE.extract_frames_kmeans_gpu(
            video_path=self.test_video,
            output_dir=output_dir,
            n_frames=5,
            step=10,
            resize_width=30
        )
        
        # Check that frames were extracted
        self.assertGreater(len(frames), 0)
        self.assertLessEqual(len(frames), 5)
        
        # Check timing information
        self.assertIn('total_time', timing)
        self.assertIn('step1_time', timing)
        self.assertIn('step2_time', timing)
        self.assertIn('step3_time', timing)
        
        # Check output directory exists
        self.assertTrue(os.path.exists(output_dir))
        
        # Check that PNG files were created
        png_files = list(Path(output_dir).glob('*.png'))
        self.assertEqual(len(png_files), len(frames))


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestClustering))
    suite.addTests(loader.loadTestsFromTestCase(TestPackageImport))
    suite.addTests(loader.loadTestsFromTestCase(TestExtraction))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
