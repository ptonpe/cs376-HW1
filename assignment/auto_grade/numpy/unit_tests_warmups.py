import unittest
import numpy as np

class TestNumpyWarmups(unittest.TestCase):
    """Simple dummy test for numpy/warmups.py to verify autograder is working."""
    
    def test_dummy_test(self):
        """Dummy test that always passes to verify autograder functionality."""
        # Simple test that checks basic numpy functionality
        arr = np.array([1, 2, 3])
        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(arr, expected)
        
        # Test that we can import the student's code
        try:
            import warmups
            self.assertTrue(True, "Successfully imported warmups module")
        except ImportError:
            self.fail("Could not import warmups module")

if __name__ == '__main__':
    unittest.main() 