#!/usr/bin/env python3
"""
Run the optimization multiple times and compare results
"""

import pandas as pd
from transport_allocation import CampVehicleAllocatorILP

def run_multiple_optimizations(num_runs=5):
    """Run optimization multiple times and compare results."""
    print(f"Running optimization {num_runs} times to compare results...")
    
    results = []
    allocator = CampVehicleAllocatorILP()
    allocator.load_data()
    
    for i in range(num_runs):
        print(f"\n{'='*60}")
        print(f"RUN {i+1}/{num_runs}")
        print(f"{'='*60}")
        
        allocation, model = allocator.solve_ilp_allocation(time_limit=600)
        if allocation:
            metrics = allocator.calculate_metrics(allocation)
            results.append({
                'run': i+1,
                'total_score': metrics['total_score'],
                'vehicles_used': metrics['vehicles_used'],
                'isolated_people': metrics['isolated_people'],
                'vehicles_with_size_2': metrics['vehicles_with_size_2'],
                'vehicles_with_size_1': metrics['vehicles_with_size_1'],
                'allocation': allocation
            })
        else:
            print(f"❌ Run {i+1} failed to find solution")
    
    # Analyze results
    if results:
        print(f"\n{'='*60}")
        print("COMPARISON OF ALL RUNS")
        print(f"{'='*60}")
        
        df = pd.DataFrame([{k: v for k, v in r.items() if k != 'allocation'} for r in results])
        print(df.to_string(index=False))
        
        # Find best result (highest score, fewest isolated people)
        best_idx = 0
        best_score = results[0]['total_score']
        fewest_isolated = results[0]['isolated_people']
        
        for i, result in enumerate(results[1:], 1):
            score = result['total_score']
            isolated = result['isolated_people']
            
            # Better if higher score, or same score with fewer isolated people
            if (score > best_score or 
                (score == best_score and isolated < fewest_isolated)):
                best_idx = i
                best_score = score
                fewest_isolated = isolated
        
        print(f"\n🏆 BEST RESULT: Run {best_idx + 1}")
        print(f"   Score: {best_score}")
        print(f"   Isolated people: {fewest_isolated}")
        print(f"   Vehicles with size 2: {results[best_idx]['vehicles_with_size_2']}")
        
        # Save the best allocation
        best_allocation = results[best_idx]['allocation']
        allocator.print_ilp_solution(best_allocation)
        allocator.export_results(best_allocation, prefix="best_allocation")
        
        return results, best_allocation
    
    return [], {}

if __name__ == "__main__":
    results, best_allocation = run_multiple_optimizations(num_runs=3)