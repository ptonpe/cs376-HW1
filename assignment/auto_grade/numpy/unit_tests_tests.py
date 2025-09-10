import unittest
import numpy as np

class TestNumpyTests(unittest.TestCase):
    """Simple dummy test for numpy/tests.py to verify autograder is working."""
    
    def test_dummy_test(self):
        """Dummy test that always passes to verify autograder functionality."""
        # Simple test that checks basic numpy functionality
        arr = np.array([1, 2, 3])
        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(arr, expected)
        
        # Test that we can import the student's code
        try:
            import tests
            self.assertTrue(True, "Successfully imported tests module")
        except ImportError:
            self.fail("Could not import tests module")

if __name__ == '__main__':
    unittest.main()