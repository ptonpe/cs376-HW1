# HW1 Autograder

This autograder is designed to test student submissions for HW1 - Numpy and Image Processing.

## How to Run

### For Testing Your Own Code

To test the code in the `code/` directory:

```bash
cd hw1/assignment
python auto_grade/autograder.py --submission_dir_name . --submission_id code --config auto_grade/config.json
```

### Command Breakdown

- `--submission_dir_name .` - Look in the current directory for submissions
- `--submission_id code` - Process the submission named "code" (your code directory)
- `--config auto_grade/config.json` - Use the configuration file in the auto_grade directory

### What Happens

1. The autograder loads the configuration from `config.json`
2. It validates that your `code/` directory contains all required files:
   - `numpy/warmups.py`
   - `numpy/tests.py` 
   - `dither/dither.py`
3. It copies test files from `auto_grade/` to your `code/` directory
4. It runs all tests and generates scores 
5. Results are saved to `grades.csv` and detailed results in `detailed_results/`

### Required Files

Your `code/` directory must contain:
- `numpy/warmups.py` - Numpy warmup functions
- `numpy/tests.py` - Numpy test functions  
- `dither/dither.py` - Dithering implementation

### Test Files

The autograder will copy these test files to your code directory:
- `numpy/unit_tests_tests.py` 
- `numpy/unit_tests_warmups.py`
- `dither/unit_tests_3_1.py`  
- `dither/unit_tests_3_2.py`  
- `dither/unit_tests_3_3.py`  
- `dither/unit_tests_3_4.py`  
- `dither/unit_tests_3_5.py`

### Output

- `grades.csv` - Summary scores for all test suites
- `detailed_results/` - Individual test results for each test suite

## Troubleshooting

- **"Config file not found"**: Make sure to use `--config auto_grade/config.json`
- **"Missing file" errors**: Ensure all required files exist in your `code/` directory
- **Test failures**: Check the detailed error messages to see which specific tests failed
- **Tests not updating**: Remove `__pycache__` directories to force Python to reload test files

## Test Structure

Each test file contains a simple dummy test that:
1. Verifies basic numpy functionality
2. Checks that your code modules can be imported
3. Calls your functions with simple inputs to verify input/output types
4. Ensures the autograder infrastructure is functioning correctly

When you're ready to implement the actual homework functions, you can replace these dummy tests with real test cases that check your implementations.


### Submission Format

Your submission folder should have the following structure:

```
your_UT_EID
└── code/                  # Student's implementation code
    ├── numpy/
    │   ├── warmups.py    # Numpy warmup functions
    │   ├── tests.py      # Numpy test functions
    │   ├── common.py     # Common utilities (MUST BE INCLUDED if you made any changes)
    │   └── run.py        # Main execution script (MUST BE INCLUDED if you made any changes)
    ├── dither/
    │   ├── dither.py     # Dithering implementation
    │   └── mychoice.jpg  # Optional - extra credit  
    ├── visualize/        # Visualization module
    │   └── mystery_visualize.py  # Mystery visualization script
    └── rubik/
        ├── info.txt
        ├── im1.jpg
        └── im2.jpg
```

**Note: You do NOT need to include any of the autograde code in your submission - only your implementation files in the `code/` directory.**

Ensure all required files from the assignment are present in the `code/` directory and test your submission locally using the autograder before submitting.



