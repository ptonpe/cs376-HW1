#!/usr/bin/env python3
"""
Generic autograder for CS376 assignments.

- Copies required files to a test environment for each submission.
- Copies test files from the tests/ directory to each submission.
- Runs unit tests based on configuration file.
- Aggregates scores and errors, outputs a CSV.
- Saves detailed test results for individual test analysis.

Directory Structure:
    auto_grade/
    ├── autograder.py          # Main autograder script
    ├── config.json            # Configuration file
    ├── detailed_results/      # Generated detailed test results
    └── grades.csv            # Generated main grades file

Usage:
    python autograder.py --submission_dir_name <dir> [--debug] [--submission_id <id>] [--config <config_file>]
"""

import sys
import os
import argparse
import unittest
import json
from typing import List, Tuple, Dict, Any, NamedTuple
import shutil
import glob

# ---------------------- Configuration Classes ----------------------

class TestConfig(NamedTuple):
    """Configuration for a single test file."""
    test_file: str  # e.g., "unit_tests_1_1.py"
    points: int
    description: str

# Global configuration loaded from config file
CONFIG = None
REQUIRED_FILES = []
OPTIONAL_FILES = []
TESTS = []

def load_config(config_file: str = "config.json"):
    """Load configuration from JSON file."""
    global CONFIG, REQUIRED_FILES, OPTIONAL_FILES, TESTS
    
    try:
        with open(config_file, 'r') as f:
            CONFIG = json.load(f)
        
        REQUIRED_FILES = CONFIG.get("required_files", [])
        OPTIONAL_FILES = CONFIG.get("optional_files", [])
        TESTS = CONFIG.get("tests", [])
        
        print(f"Loaded configuration: {CONFIG.get('assignment_name', 'Unknown Assignment')}")
        print(f"Required files: {REQUIRED_FILES}")
        print(f"Optional files: {OPTIONAL_FILES}")
        print(f"Tests: {len(TESTS)} test files configured")
        
    except FileNotFoundError:
        print(f"Error: Config file '{config_file}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file: {e}")
        sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", help="Debug mode", action="store_true")
    parser.add_argument("--submission_id", help="Submission id", type=str, default=None)
    parser.add_argument("--submission_dir_name", default="code", help="Path to students submission folder")
    parser.add_argument("--save_results", help="Path to save the results", default="auto_grade/grades.csv")
    parser.add_argument("--no_comments", help="Exclude comments from CSV output", action="store_true")
    parser.add_argument("--config", help="Configuration file path", default="config.json")
    args = parser.parse_args()
    return args

# ---------------------- Helper Functions ----------------------

def find_student_dirs(submission_path: str, dir_name: str) -> List[str]:
    """Find directories with given name in submission, excluding __MACOSX and hidden files."""
    found_dirs = []
    for root, dirs, files in os.walk(submission_path):
        # Skip __MACOSX and hidden directories
        dirs[:] = [d for d in dirs if d != '__MACOSX' and not d.startswith('.')]
        if dir_name in dirs:
            found_dirs.append(os.path.join(root, dir_name))
    return found_dirs

# ---------------------- Submission Structure Validation ----------------------

def validate_submission_structure(submission_path: str) -> Tuple[bool, str]:
    """Check if all required files exist in the submission."""
    missing = []
    
    for file_pattern in REQUIRED_FILES:
        if len(glob.glob(os.path.join(submission_path+"/**", file_pattern), recursive=True)) == 0:
            missing.append(f"Missing file: {file_pattern}")
    if missing:
        return False, "; ".join(missing)
    return True, ""

def find_submission_code_dir(submission_path: str) -> str:
    """
    Find the directory containing any file from REQUIRED_FILES or OPTIONAL_FILES.
    Prioritizes directories with required code files.
    """
    full_path = glob.glob(os.path.join(submission_path+"/**", REQUIRED_FILES[0]), recursive=True)
    root = os.path.dirname(full_path[0].split(REQUIRED_FILES[0])[0])    
    return root

def copy_missing_files(submission_path: str) -> str:
    """Copy missing optional files from the tests directory to submission."""
    errors = []
    
    # Find the directory where student's code is located
    code_dir = find_submission_code_dir(submission_path)

    # Copy missing files to the code directory
    for file_pattern in OPTIONAL_FILES:
        # Get the autograder directory to find the tests folder
        autograder_dir = os.path.dirname(os.path.abspath(__file__))
        source_path = os.path.join(autograder_dir, file_pattern)
        
        # Check if file already exists in the target directory
        if os.path.exists(os.path.join(code_dir, file_pattern)):
            continue
        
        # Copy file to the target directory
        target_file = os.path.join(code_dir, file_pattern)
        try:
            if os.path.exists(source_path):
                shutil.copy2(source_path, target_file)
            else:
                errors.append(f"Source file {source_path} not found")
        except Exception as e:
            errors.append(f"Failed to copy {file_pattern}: {str(e)}")
    
    return "; ".join(errors)

# ---------------------- Test Running Utilities ----------------------

def get_score_from_unittest_clean(test_file_path: str, test_name: str = "Unit tests", points: int = 100) -> Tuple[str, int, Dict[str, int]]:
    """
    Run unit tests and return errors, score, and detailed individual test results.
    Now takes the full test file path instead of pattern and submission_dir.
    Returns: (errors, score, individual_test_results)
    """
    # Extract directory and filename from test_file_path
    test_dir = os.path.dirname(test_file_path)
    test_file = os.path.basename(test_file_path)
    
    # Import and run tests programmatically
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern=test_file)
    
    # Collect all test method names from the suite before running
    all_test_names = set()
    
    def collect_test_names(test_suite):
        """Recursively collect all test method names from a test suite."""
        for test in test_suite:
            if hasattr(test, '_testMethodName'):
                # This is a test method
                all_test_names.add(test._testMethodName)
            elif hasattr(test, '_tests'):
                # This is a test suite, recurse
                collect_test_names(test._tests)
            elif hasattr(test, 'countTestCases') and test.countTestCases() > 0:
                # This might be a test case, try to get its method name
                if hasattr(test, '_testMethodName'):
                    all_test_names.add(test._testMethodName)
    
    collect_test_names(suite)
    
    # Suppress all output during test run
    with open(os.devnull, 'w') as devnull:
        runner = unittest.TextTestRunner(stream=devnull)
        result = runner.run(suite)
    
    # Calculate score directly from result object
    total_tests = result.testsRun
    failed_tests = len(result.failures) + len(result.errors)
    passed_tests = total_tests - failed_tests
    
    # Create detailed individual test results
    individual_test_results = {}
    failed_test_names = set()
    
    # Mark failed tests
    for test, traceback in result.failures + result.errors:
        failed_test_names.add(test._testMethodName)
    
    # Create test results using actual test method names
    if total_tests > 0:
        # First, mark failed tests (we know their names)
        for test_name in failed_test_names:
            individual_test_results[test_name] = 0
        
        # Then mark passed tests using actual test method names
        for test_name in all_test_names:
            if test_name not in failed_test_names:
                individual_test_results[test_name] = 1
        
    # Get failed test names for error string
    failed_names = list(failed_test_names)
    
    score = int((passed_tests / total_tests) * points) if total_tests > 0 else 0
    
    errors = ""
    if failed_names != []:
        errors = f"{' '.join(failed_names)} "
    
    return errors, score, individual_test_results

def add_submission_path_to_sys_path(submission_path):
    
    # Store original sys.path
    original_path = sys.path.copy()
    
    # Add the submission path to sys.path
    sys.path.insert(0, submission_path)
    
    # Store modules that existed before this test
    original_modules = set(sys.modules.keys())
    
    return original_path, original_modules

def remove_submission_path_from_sys_path(original_path, original_modules, submission_path):
    # Restore original sys.path
    sys.path = original_path
    
    # Remove modules that were added during this submission's testing
    current_modules = set(sys.modules.keys())
    for module_name in list(current_modules):
        if module_name not in original_modules:
            mod = sys.modules.get(module_name)
            # Only delete if it's from the current submission's path
            if hasattr(mod, '__file__') and mod.__file__:
                module_path = os.path.abspath(mod.__file__)
                submission_abs_path = os.path.abspath(submission_path)
                if submission_abs_path in module_path:
                    del sys.modules[module_name]

# ---------------------- Generic Scoring Functions ----------------------

def score_tests(submission_path: str) -> Tuple[List[int], List[str], Dict[str, Dict[str, int]]]:
    """
    Generic function to score all tests based on configuration.
    Returns individual test scores, errors for each test, and detailed test results.
    """
    test_scores = []
    test_errors = []
    detailed_test_results = {}
    
    # Find the directory where student's code is located
    code_dir = find_submission_code_dir(submission_path)
    
    # Run each test file
    for test_config in TESTS:
        # Find the test file in the code directory
        test_file_name = test_config["test_file"]
        test_file_path = os.path.join(code_dir, test_file_name)
        
        if not os.path.exists(test_file_path):
            test_errors.append(f"{test_file_name} not found")
            test_scores.append(0)
            detailed_test_results[test_config["description"]] = {}
            continue
        
        # Run the test
        test_error, test_score, individual_test_results = get_score_from_unittest_clean(
            test_file_path,
            test_name=test_config["description"],
            points=test_config["points"]
        )
        test_scores.append(test_score)
        test_errors.append(test_error)
        detailed_test_results[test_config["description"]] = individual_test_results
    
    # Print detailed breakdown
    print("Test detailed breakdown:")
    for test_config, test_score in zip(TESTS, test_scores):
        print(f"  {test_config['description']}: {test_score}/{test_config['points']}")
    
    return test_scores, test_errors, detailed_test_results

# ---------------------- Detailed Results Saving ----------------------

def save_detailed_test_results(submission_id: str, detailed_results: Dict[str, Dict[str, int]], args: argparse.Namespace):
    """
    Save detailed test results for a student. This function accumulates data that will be written 
    to test-file-based CSV files at the end of all submissions processing.
    """
    # Store the results in a global structure for later CSV writing
    if not hasattr(save_detailed_test_results, 'all_results'):
        save_detailed_test_results.all_results = {}
    
    # Store results organized by test description
    for test_description, test_results in detailed_results.items():
        if test_description not in save_detailed_test_results.all_results:
            save_detailed_test_results.all_results[test_description] = {}
        
        save_detailed_test_results.all_results[test_description][submission_id] = test_results

def write_test_csv_files(args: argparse.Namespace):
    """
    Write CSV files for each test showing all students' individual test results.
    Each CSV has students as rows and individual test cases as columns.
    """
    if not hasattr(save_detailed_test_results, 'all_results'):
        return
    
    # Create results directory if it doesn't exist
    # Always create detailed_results in the same directory as the autograder
    autograder_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(autograder_dir, "detailed_results")
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = save_detailed_test_results.all_results
    
    # Create one CSV file per test file (subquestion)
    for test_section, test_data in all_results.items():
        csv_file = os.path.join(results_dir, f"{test_section}_detailed_results.csv")
        
        # Collect all unique test case names across all students for this test file
        all_test_names = set()
        for student_id, test_results in test_data.items():
            for test_name in test_results.keys():
                all_test_names.add(test_name)
        
        # Sort test names numerically for consistent column order
        # This ensures test_1, test_2, test_3... instead of test_1, test_10, test_11, test_2...
        def natural_sort_key(test_name):
            if test_name.startswith('test_'):
                try:
                    # Handle cases like test_1, test_2, test_3
                    if test_name.count('_') == 1:
                        return (0, int(test_name.split('_')[1]))  # Use tuple for consistent sorting
                    # Handle cases like test_patch_size_2x2, test_patch_size_4x4
                    elif test_name.count('_') >= 2:
                        # Extract the last part and try to convert to int
                        last_part = test_name.split('_')[-1]
                        try:
                            return (1, int(last_part))  # Use tuple for consistent sorting
                        except ValueError:
                            # If last part isn't a number, use the original string
                            return (2, test_name)
                except (ValueError, IndexError):
                    return (2, test_name)
            return (2, test_name)
        
        sorted_test_names = sorted(all_test_names, key=natural_sort_key)
        
        # Write CSV file
        with open(csv_file, "w") as f:
            # Write header
            f.write("student_id," + ",".join(sorted_test_names) + "\n")
            
            # Write data for each student
            for student_id in sorted(test_data.keys()):
                row_data = [student_id]
                
                for test_name in sorted_test_names:
                    # Get the result for this test (0 or 1)
                    result = ""
                    if test_name in test_data[student_id]:
                        result = str(test_data[student_id][test_name])
                    
                    row_data.append(result)
                
                f.write(",".join(row_data) + "\n")

# ---------------------- Main Flow ----------------------

def process_submission(submission: str, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Process a single submission and return results.
    """
    # Handle submission ID parsing - try to extract from name or use the whole name
    if "_" in submission and len(submission.split("_")) > 1:
        submission_id = submission.split("_")[1]
    else:
        submission_id = submission
        
    submission_path = os.path.join(args.submission_dir_name, submission)
    

    # Validate structure before scoring
    print(f"Validating submission path: {submission_path}")
    is_valid, structure_error = validate_submission_structure(submission_path)
    if not is_valid:
        print(f"Submission {submission_id} failed structure check: {structure_error}")
        result = {"submission_id": submission_id}
        for test_config in TESTS:
            test_name = test_config["description"].replace(" ", "_").replace("(", "").replace(")", "").replace(".", "").lower()
            result[f"score_{test_name}"] = 0
            result[f"errors_{test_name}"] = f"Structure error: {structure_error}"
        return result
    else:
        print(f"Submission {submission_id} structure validation passed")
    
    # Copy missing optional files
    copy_errors = copy_missing_files(submission_path)
    if copy_errors:
        print(f"Warning for submission {submission_id}: {copy_errors}")
    else:
        print(f"Submission {submission_id} file copying completed successfully")
    
    # Add submission path to sys.path for this submission
    original_path, original_modules = add_submission_path_to_sys_path(submission_path)
    
    try:
        result = {"submission_id": submission_id}
        detailed_results = {}
        
        # Score all tests using the generic function
        test_scores, test_errors, detailed_test_results = score_tests(submission_path)
        print(f"Test scores: {test_scores}")
        print(f"Test errors: {test_errors}")
        
        # Store detailed test results
        detailed_results = detailed_test_results
        
        # Store individual test scores and errors
        for i, (test_config, score, error) in enumerate(zip(TESTS, test_scores, test_errors)):
            test_name = test_config["description"].replace(" ", "_").replace("(", "").replace(")", "").replace(".", "").lower()
            result[f"score_{test_name}"] = score
            result[f"errors_{test_name}"] = error
        
        # Save detailed test results for this student
        save_detailed_test_results(submission_id, detailed_results, args)
        
        return result
    finally:
        # Always remove submission path from sys.path, even if there's an error
        remove_submission_path_from_sys_path(original_path, original_modules, submission_path)

def main():
    """
    Main entry point for the autograder.
    """
    args = parse_args()
    
    # Load configuration
    load_config(args.config)
    
    # Use the submission directory name as provided, resolving relative to current directory
    submission_dir_name = os.path.abspath(args.submission_dir_name)
    
    csv_data = []
    
    # Generate CSV header dynamically from tests configuration
    csv_header = ["submission_id"]
    for test_config in TESTS:
        test_name = test_config["description"].replace(" ", "_").replace("(", "").replace(")", "").replace(".", "").lower()
        csv_header.append(f"score_{test_name}")
        if not args.no_comments:
            csv_header.append(f"errors_{test_name}")
    
    for submission in os.listdir(submission_dir_name):
        
        ## if submission is not a directory, skip
        if not os.path.isdir(os.path.join(submission_dir_name, submission)):
            continue
        
        # Check if this is the submission we want
        # Use the directory name as the submission ID directly
        submission_id = submission
            
        if args.submission_id is not None and submission_id != args.submission_id:
            continue
        
        print("---------------------------------------------------")
        result = process_submission(submission, args)
        
        # Create CSV row dynamically
        csv_row = [result["submission_id"]]
        for test_config in TESTS:
            test_name = test_config["description"].replace(" ", "_").replace("(", "").replace(")", "").replace(".", "").lower()
            csv_row.append(result[f"score_{test_name}"])
            if not args.no_comments:
                csv_row.append(result[f"errors_{test_name}"])
        
        csv_data.append(csv_row)
            
        print("---------------------------------------------------")
        if args.submission_id is not None:
            break
        
    ## save the csv file to the submission path
    save_path = os.path.join(os.getcwd(), args.save_results)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(",".join(csv_header) + "\n")
        for data in csv_data:
            f.write(",".join(str(item) for item in data) + "\n")
    
    ## save detailed test results as test-based CSV files
    write_test_csv_files(args)
        
if __name__ == '__main__':
    main()
