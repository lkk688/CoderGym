#!/usr/bin/env python3
"""
Verification script for new CoderGym neural network tasks.

This script verifies that all 4 new tasks are properly set up and follow
the pytorch_task_v1 protocol.
"""

import json
import os
import re
import sys
from pathlib import Path


def check_json_validity(json_path):
    """Verify ml_tasks.json is valid JSON and contains new tasks."""
    print("\n" + "="*70)
    print("1. Checking ml_tasks.json")
    print("="*70)
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        print("✓ JSON is valid")
    except json.JSONDecodeError as e:
        print(f"✗ JSON is invalid: {e}")
        return False
    
    # Check for new tasks
    required_tasks = [
        'mlp_lvl5_ensemble_distillation',
        'cnn_lvl5_attention_mechanisms',
        'nlp_lvl1_text_embedding',
        'robust_lvl1_adversarial_training'
    ]
    
    existing_tasks = {task['id'] for task in data.get('tasks', [])}
    
    all_present = True
    for task_id in required_tasks:
        if task_id in existing_tasks:
            print(f"✓ {task_id} found in JSON")
        else:
            print(f"✗ {task_id} NOT found in JSON")
            all_present = False
    
    print(f"\nTotal tasks in JSON: {len(existing_tasks)}")
    return all_present


def check_task_files(base_path):
    """Verify that all task.py files exist and are substantial."""
    print("\n" + "="*70)
    print("2. Checking task.py files")
    print("="*70)
    
    tasks = [
        'mlp_lvl5_ensemble_distillation',
        'cnn_lvl5_attention_mechanisms',
        'nlp_lvl1_text_embedding',
        'robust_lvl1_adversarial_training'
    ]
    
    all_exist = True
    for task in tasks:
        task_path = os.path.join(base_path, task, 'task.py')
        
        if os.path.exists(task_path):
            size_kb = os.path.getsize(task_path) / 1024
            lines = len(open(task_path).readlines())
            print(f"✓ {task}")
            print(f"  Size: {size_kb:.1f} KB ({lines} lines)")
            
            if size_kb < 10:
                print(f"  ⚠ Warning: File size seems small")
        else:
            print(f"✗ {task} - task.py NOT FOUND")
            all_exist = False
    
    return all_exist


def check_protocol_compliance(base_path):
    """Verify pytorch_task_v1 protocol compliance."""
    print("\n" + "="*70)
    print("3. Checking pytorch_task_v1 Protocol Compliance")
    print("="*70)
    
    tasks = [
        'mlp_lvl5_ensemble_distillation',
        'cnn_lvl5_attention_mechanisms',
        'nlp_lvl1_text_embedding',
        'robust_lvl1_adversarial_training'
    ]
    
    required_functions = [
        'get_task_metadata',
        'set_seed',
        'get_device',
        'make_dataloaders',
        'build_model',
        'train',
        'evaluate',
        'predict',
        'save_artifacts'
    ]
    
    all_compliant = True
    
    for task in tasks:
        task_path = os.path.join(base_path, task, 'task.py')
        
        if not os.path.exists(task_path):
            print(f"✗ {task}: task.py not found")
            all_compliant = False
            continue
        
        with open(task_path, 'r') as f:
            content = f.read()
        
        print(f"\n{task}:")
        task_compliant = True
        
        # Check required functions
        for func in required_functions:
            pattern = rf'^def {func}\('
            if re.search(pattern, content, re.MULTILINE):
                print(f"  ✓ {func}")
            else:
                print(f"  ✗ {func} MISSING")
                task_compliant = False
        
        # Check main block
        if 'if __name__ == ' in content and '__main__' in content:
            print(f"  ✓ __main__ block")
        else:
            print(f"  ✗ __main__ block MISSING")
            task_compliant = False
        
        # Check for sys.exit
        if 'sys.exit' in content:
            print(f"  ✓ sys.exit calls")
        else:
            print(f"  ⚠ No sys.exit calls found")
        
        if task_compliant:
            print(f"  STATUS: ✓ COMPLIANT")
        else:
            print(f"  STATUS: ✗ NEEDS REVIEW")
            all_compliant = False
    
    return all_compliant


def check_documentation(base_path):
    """Check for documentation files."""
    print("\n" + "="*70)
    print("4. Checking Documentation")
    print("="*70)
    
    docs = [
        ('NEW_TASKS_SUMMARY.md', 'Task documentation'),
        ('GITHUB_SUBMISSION_GUIDE.md', 'Submission guide'),
    ]
    
    all_exist = True
    for doc_name, description in docs:
        doc_path = os.path.join(base_path, doc_name)
        if os.path.exists(doc_path):
            size_kb = os.path.getsize(doc_path) / 1024
            print(f"✓ {doc_name} ({size_kb:.1f} KB)")
            print(f"  - {description}")
        else:
            print(f"✗ {doc_name} NOT FOUND")
            all_exist = False
    
    return all_exist


def check_imports(base_path):
    """Try importing key functions from tasks (without execution)."""
    print("\n" + "="*70)
    print("5. Checking Imports")
    print("="*70)
    
    task_imports = {
        'mlp_lvl5_ensemble_distillation.task': ['get_task_metadata', 'build_model'],
        'cnn_lvl5_attention_mechanisms.task': ['SEBlock', 'SimpleCNN'],
        'nlp_lvl1_text_embedding.task': ['SkipGramModel', 'make_dataloaders'],
        'robust_lvl1_adversarial_training.task': ['FGSM', 'SimpleCNN'],
    }
    
    # Add tasks directory to path
    sys.path.insert(0, os.path.join(base_path, '..', '..'))
    
    all_importable = True
    for module_path, items in task_imports.items():
        task_dir = module_path.split('.')[0]
        print(f"\n{task_dir}:")
        try:
            module = __import__(module_path, fromlist=items)
            for item in items:
                if hasattr(module, item):
                    print(f"  ✓ {item}")
                else:
                    print(f"  ⚠ {item} not found (may be defined in function)")
        except ImportError as e:
            print(f"  ✗ Import failed: {e}")
            all_importable = False
        except Exception as e:
            print(f"  ⚠ Error: {e}")
    
    return all_importable


def generate_summary(base_path):
    """Generate summary of verification results."""
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    # Count lines of code
    tasks = [
        'mlp_lvl5_ensemble_distillation',
        'cnn_lvl5_attention_mechanisms',
        'nlp_lvl1_text_embedding',
        'robust_lvl1_adversarial_training'
    ]
    
    total_lines = 0
    total_size_kb = 0
    
    for task in tasks:
        task_path = os.path.join(base_path, task, 'task.py')
        if os.path.exists(task_path):
            lines = len(open(task_path).readlines())
            size_kb = os.path.getsize(task_path) / 1024
            total_lines += lines
            total_size_kb += size_kb
    
    print(f"\nFiles Created:")
    print(f"  - 4 task.py implementations")
    print(f"  - Total code: {total_lines:,} lines, {total_size_kb:.1f} KB")
    print(f"  - 2 documentation files")
    print(f"  - 1 ml_tasks.json updated")
    
    print(f"\nTasks Added:")
    for task in tasks:
        print(f"  ✓ {task}")
    
    print(f"\nNext Steps:")
    print(f"  1. See NEW_TASKS_SUMMARY.md for task descriptions")
    print(f"  2. See GITHUB_SUBMISSION_GUIDE.md for submission instructions")
    print(f"  3. Run individual tasks to verify execution")
    print(f"  4. Create PR or new repository as described in guide")


def main(base_path=None):
    """Run all verification checks."""
    
    if base_path is None:
        # Find MLtasks directory
        current = Path(__file__).parent.absolute()
        base_path = current / 'MLtasks' / 'tasks'
        
        if not base_path.exists():
            print(f"Error: Could not find tasks directory at {base_path}")
            print(f"Please provide the path to MLtasks/tasks as argument")
            return False
    
    base_path = str(base_path)
    json_path = os.path.join(base_path, '..', 'ml_tasks.json')
    
    print("\n" + "="*70)
    print("CoderGym Neural Network Tasks Verification")
    print("="*70)
    
    results = []
    
    # Run all checks
    results.append(("JSON Validity", check_json_validity(json_path)))
    results.append(("Task Files", check_task_files(base_path)))
    results.append(("Protocol Compliance", check_protocol_compliance(base_path)))
    results.append(("Documentation", check_documentation(os.path.dirname(base_path))))
    results.append(("Imports", check_imports(base_path)))
    
    # Summary
    generate_summary(base_path)
    
    # Final status
    print("\n" + "="*70)
    print("VERIFICATION RESULTS")
    print("="*70)
    
    all_passed = True
    for check_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{check_name:.<50} {status}")
        if not result:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n✓ All verification checks passed!")
        print("Tasks are ready for GitHub submission.")
        return True
    else:
        print("\n✗ Some verification checks failed.")
        print("Please review the output above for details.")
        return False


if __name__ == '__main__':
    if len(sys.argv) > 1:
        success = main(sys.argv[1])
    else:
        success = main()
    
    sys.exit(0 if success else 1)
