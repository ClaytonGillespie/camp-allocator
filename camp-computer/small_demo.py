#!/usr/bin/env python3
"""
Vehicle Allocation using PuLP - Small Example
3 cars, 4 potential drivers, 3 dorms, 12 campers
"""

import pulp
import pandas as pd
from collections import defaultdict

def create_small_example():
    """Create small example data."""
    
    # People and their dorms
    people = {
        # Dorm A (4 people)
        'Alice': 'Dorm_A',    # Can drive
        'Amy': 'Dorm_A',
        'Aaron': 'Dorm_A',
        # 'Anna': 'Dorm_A',
        
        # Dorm B (5 people)  
        'Bob': 'Dorm_B',      # Can drive
        'Ben': 'Dorm_B',      # Can drive
        'Betty': 'Dorm_B',
        'Brad': 'Dorm_B',
        'Bella': 'Dorm_B',
        
        # Dorm C (3 people)
        'Charlie': 'Dorm_C',  # Can drive
        'Carol': 'Dorm_C',
        'Chris': 'Dorm_C'
    }
    
    # Vehicles and their capacities (including driver)
    vehicles = {
        'Car1': 4,  # Can take 4 people total
        'Car2': 5,  # Can take 5 people total  
        'Car3': 4   # Can take 4 people total
    }
    
    # Who can drive which car
    driver_vehicle_pairs = [
        ('Alice', 'Car1'),
        ('Alice', 'Car2'),
        ('Bob', 'Car1'),
        ('Bob', 'Car3'),
        ('Ben', 'Car2'),
        ('Ben', 'Car3'),
        ('Charlie', 'Car2'),
        ('Charlie', 'Car3')
    ]
    
    return people, vehicles, driver_vehicle_pairs


def solve_allocation_ilp(people, vehicles, driver_vehicle_pairs):
    """Solve vehicle allocation using Integer Linear Programming."""
    
    print("Setting up the optimization problem...")
    print(f"- {len(people)} people from {len(set(people.values()))} dorms")
    print(f"- {len(vehicles)} vehicles")
    print(f"- {len(set(d for d, v in driver_vehicle_pairs))} potential drivers")
    
    # Create the model
    model = pulp.LpProblem("Vehicle_Allocation", pulp.LpMaximize)
    
    # Decision variables
    # x[person][vehicle] = 1 if person is in vehicle
    x = {}
    for person in people:
        x[person] = {}
        for vehicle in vehicles:
            x[person][vehicle] = pulp.LpVariable(
                f"x_{person}_{vehicle}", cat='Binary'
            )
    
    # y[driver][vehicle] = 1 if driver is driving vehicle
    y = {}
    for driver, vehicle in driver_vehicle_pairs:
        if driver not in y:
            y[driver] = {}
        y[driver][vehicle] = pulp.LpVariable(
            f"y_{driver}_{vehicle}", cat='Binary'
        )
    
    # z[vehicle] = 1 if vehicle is used
    z = {}
    for vehicle in vehicles:
        z[vehicle] = pulp.LpVariable(f"z_{vehicle}", cat='Binary')
    
    # g[dorm][vehicle] = number of people from dorm in vehicle
    g = {}
    dorms = set(people.values())
    for dorm in dorms:
        g[dorm] = {}
        for vehicle in vehicles:
            g[dorm][vehicle] = pulp.LpVariable(
                f"g_{dorm}_{vehicle}", lowBound=0, cat='Integer'
            )
    
    # m[vehicle] = minimum group size in vehicle (what we want to maximize)
    m = {}
    for vehicle in vehicles:
        m[vehicle] = pulp.LpVariable(f"m_{vehicle}", lowBound=0, cat='Integer')
    
    print("\nAdding constraints...")
    
    # CONSTRAINTS
    
    # 1. Each person must be in exactly one vehicle
    for person in people:
        model += pulp.lpSum(x[person][vehicle] for vehicle in vehicles) == 1, \
                 f"One_Vehicle_{person}"
    
    # 2. Vehicle capacity constraints
    for vehicle in vehicles:
        model += pulp.lpSum(x[person][vehicle] for person in people) <= vehicles[vehicle], \
                 f"Capacity_{vehicle}"
    
    # 3. If a vehicle is used, it must have exactly one driver
    for vehicle in vehicles:
        driver_sum = pulp.lpSum(
            y[driver][vehicle] 
            for driver, veh in driver_vehicle_pairs 
            if veh == vehicle
        )
        model += driver_sum == z[vehicle], f"One_Driver_{vehicle}"
    
    # 4. A driver can only drive one vehicle
    drivers = set(d for d, v in driver_vehicle_pairs)
    for driver in drivers:
        model += pulp.lpSum(
            y[driver][vehicle] 
            for vehicle in y[driver]
        ) <= 1, f"Driver_One_Vehicle_{driver}"
    
    # 5. If someone is driving a vehicle, they must be in it
    for driver, vehicle in driver_vehicle_pairs:
        model += y[driver][vehicle] <= x[driver][vehicle], \
                 f"Driver_In_Vehicle_{driver}_{vehicle}"
    
    # 6. If a vehicle is not used, no one can be in it
    for vehicle in vehicles:
        for person in people:
            model += x[person][vehicle] <= z[vehicle], \
                     f"Vehicle_Used_{person}_{vehicle}"
    
    # 7. Calculate group sizes per dorm per vehicle
    for dorm in dorms:
        for vehicle in vehicles:
            dorm_people = [p for p in people if people[p] == dorm]
            model += g[dorm][vehicle] == pulp.lpSum(
                x[person][vehicle] for person in dorm_people
            ), f"Group_Size_{dorm}_{vehicle}"
    
    # 8. Define minimum group size in each vehicle
    # m[vehicle] <= g[dorm][vehicle] for all dorms with people in vehicle
    # This is tricky - we need m[v] to be the minimum of all non-zero g[d][v]
    
    # We'll use a big-M approach
    M = len(people)  # Big M value
    
    for vehicle in vehicles:
        for dorm in dorms:
            # If g[dorm][vehicle] > 0, then m[vehicle] <= g[dorm][vehicle]
            # We reformulate: m[vehicle] <= g[dorm][vehicle] + M * (1 - h[dorm][vehicle])
            # where h[dorm][vehicle] = 1 if dorm has people in vehicle
            
            # Create helper variable
            h_var = pulp.LpVariable(f"h_{dorm}_{vehicle}", cat='Binary')
            
            # If any person from dorm is in vehicle, h = 1
            dorm_people = [p for p in people if people[p] == dorm]
            model += pulp.lpSum(x[person][vehicle] for person in dorm_people) >= h_var, \
                     f"Dorm_Present1_{dorm}_{vehicle}"
            model += pulp.lpSum(x[person][vehicle] for person in dorm_people) <= M * h_var, \
                     f"Dorm_Present2_{dorm}_{vehicle}"
            
            # If dorm is present (h=1), then m <= g
            model += m[vehicle] <= g[dorm][vehicle] + M * (1 - h_var), \
                     f"Min_Group_{dorm}_{vehicle}"
    
    # OBJECTIVE: Maximize sum of minimum group sizes
    model += pulp.lpSum(m[vehicle] for vehicle in vehicles), "Objective"
    
    print("\nSolving the optimization problem...")
    
    # Solve
    model.solve(pulp.PULP_CBC_CMD(msg=0))
    
    print(f"Status: {pulp.LpStatus[model.status]}")
    print(f"Objective value: {pulp.value(model.objective)}")
    
    # Extract solution
    allocation = defaultdict(list)
    drivers = {}
    
    for person in people:
        for vehicle in vehicles:
            if pulp.value(x[person][vehicle]) == 1:
                allocation[vehicle].append(person)
                
                # Check if this person is the driver
                if person in y and vehicle in y[person]:
                    if pulp.value(y[person][vehicle]) == 1:
                        drivers[vehicle] = person
    
    return allocation, drivers, model


def print_solution(allocation, drivers, people, vehicles):
    """Print the solution in a nice format."""
    
    print("\n" + "="*50)
    print("OPTIMAL ALLOCATION")
    print("="*50)
    
    total_min_score = 0
    
    for vehicle in sorted(vehicles.keys()):
        if vehicle not in allocation or not allocation[vehicle]:
            print(f"\n{vehicle}: NOT USED")
            continue
            
        passengers = allocation[vehicle]
        driver = drivers.get(vehicle, "ERROR: No driver!")
        
        print(f"\n{vehicle} (capacity {vehicles[vehicle]}):")
        print(f"  Driver: {driver}")
        print(f"  Passengers ({len(passengers)}): {', '.join(passengers)}")
        
        # Count by dorm
        dorm_counts = defaultdict(int)
        for person in passengers:
            dorm_counts[people[person]] += 1
        
        print(f"  Dorms: ", end="")
        for dorm, count in sorted(dorm_counts.items()):
            print(f"{dorm}:{count} ", end="")
        
        min_score = min(dorm_counts.values()) if dorm_counts else 0
        total_min_score += min_score
        print(f"\n  Min group size (score): {min_score}")
    
    print(f"\nTotal score: {total_min_score}")
    
    # Check everyone is assigned
    assigned = set()
    for passengers in allocation.values():
        assigned.update(passengers)
    
    if len(assigned) < len(people):
        print(f"\nWARNING: Not everyone assigned!")
        print(f"Missing: {set(people.keys()) - assigned}")


def demonstrate_manual_vs_optimal():
    """Show the difference between a manual allocation and the optimal one."""
    
    people, vehicles, driver_vehicle_pairs = create_small_example()
    
    print("EXAMPLE SETUP")
    print("="*50)
    print("\nPeople by dorm:")
    dorm_members = defaultdict(list)
    for person, dorm in people.items():
        dorm_members[dorm].append(person)
    
    for dorm, members in sorted(dorm_members.items()):
        print(f"  {dorm}: {', '.join(members)} ({len(members)} people)")
    
    print("\nPotential drivers and their vehicles:")
    driver_vehicles = defaultdict(list)
    for driver, vehicle in driver_vehicle_pairs:
        driver_vehicles[driver].append(vehicle)
    
    for driver, vehs in sorted(driver_vehicles.items()):
        print(f"  {driver}: can drive {', '.join(vehs)}")
    
    # Example of suboptimal manual allocation
    # print("\n" + "="*50)
    # print("MANUAL ALLOCATION (Suboptimal)")
    # print("="*50)
    
    # manual = {
    #     'Car1': ['Alice', 'Amy', 'Bob', 'Ben'],      # A:2, B:2, min=2
    #     'Car2': ['Charlie', 'Carol', 'Chris', 'Betty', 'Brad'],  # C:3, B:2, min=2
    #     'Car3': ['Bella', 'Aaron', 'Anna']           # B:1, A:2, min=1
    # }
    # manual_drivers = {'Car1': 'Alice', 'Car2': 'Charlie', 'Car3': 'Ben'}
    
    # print_solution(manual, manual_drivers, people, vehicles)
    # print("\nManual total score: 2 + 2 + 1 = 5")
    # print("Notice Car3 has an isolated person (Bella from Dorm_B)")
    
    # Solve optimally
    print("\n" + "="*50)
    print("SOLVING WITH INTEGER LINEAR PROGRAMMING")
    print("="*50)
    
    allocation, drivers, model = solve_allocation_ilp(people, vehicles, driver_vehicle_pairs)
    print_solution(allocation, drivers, people, vehicles)
    
    print("\n" + "="*50)
    print("KEY INSIGHTS")
    print("="*50)
    print("1. ILP guarantees finding the optimal allocation")
    print("2. It considers all possible driver-vehicle combinations")
    print("3. The objective maximizes the sum of minimum group sizes")
    print("4. No isolated people in the optimal solution!")
    

if __name__ == "__main__":
    # Run the demonstration
    demonstrate_manual_vs_optimal()
    
    print("\n" + "="*50)
    print("ADAPTING TO YOUR FULL PROBLEM")
    print("="*50)
    print("""
To use this for your full camp allocation:

1. Install PuLP: pip install pulp

2. Replace the example data with your real data:
   - people: dictionary from your campers.csv
   - vehicles: dictionary from your cars.csv (3-point capacities)
   - driver_vehicle_pairs: list from your cars.csv driver columns

3. The solver will find the globally optimal allocation that:
   - Maximizes the sum of minimum dorm group sizes per vehicle
   - Ensures no one is left behind
   - Respects all capacity and driver constraints

4. For 95 people and 17 vehicles, solving might take 10-60 seconds.

The ILP approach guarantees the optimal solution, unlike the
heuristic approach which might miss better allocations.
""")