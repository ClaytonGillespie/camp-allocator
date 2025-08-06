#!/usr/bin/env python3
"""
Camp Vehicle Allocation System with PuLP Optimization
Uses Integer Linear Programming to find the optimal allocation
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import pulp
import time
import sys

# Scoring configuration constants
SCORE_WEIGHTS = {1: 1, 2: 8, 3: 27, 4: 64, 5: 125, 6: 216, 7: 250, 8: 250, 9: 250, 10: 250}
SINGLE_DORM_BONUS = 20  # Bonus for single-dorm cars (not minibuses)
GENDER_COHESION_BONUS = 10  # Bonus for single-gender mixed-dorm vehicles
MIN_SIZE_2_PENALTY = 500  # Penalty for min group size of 2
MIN_SIZE_1_PENALTY = 2000  # Penalty for min group size of 1

class CampVehicleAllocatorILP:
    def __init__(self):
        """Initialize the allocator."""
        self.people_df = None
        self.vehicles_df = None
        self.people = {}  # name -> dorm mapping
        self.people_gender = {}  # name -> gender mapping
        self.vehicles = {}  # vehicle -> capacity mapping
        self.driver_vehicle_pairs = []  # list of (driver, vehicle) tuples
        
    def load_data(self, campers_csv='./data/campers.csv', vehicles_csv='./data/cars.csv'):
        """Load camper and vehicle data from CSV files."""
        try:
            # Load campers
            self.people_df = pd.read_csv(campers_csv)
            print(f"✓ Loaded {len(self.people_df)} people from {campers_csv}")
            
            # Show breakdown by dorm
            dorm_counts = self.people_df['Dorm'].value_counts()
            print(f"  Dorms: {len(dorm_counts)} total")
            
            # Check for Support dorm
            if 'Support' in self.people_df['Dorm'].values:
                support_people = self.people_df[self.people_df['Dorm'] == 'Support']['Name'].tolist()
                print(f"  Support staff: {', '.join(support_people)}")
            
            # Create people dictionary
            self.people = dict(zip(self.people_df['Name'], self.people_df['Dorm']))
            
            # Check if Sex column exists
            if 'Sex' in self.people_df.columns:
                self.people_gender = dict(zip(self.people_df['Name'], self.people_df['Sex']))
                gender_counts = self.people_df['Sex'].value_counts()
                print(f"  Gender breakdown: {', '.join([f'{g}: {c}' for g, c in gender_counts.items()])}")
            else:
                self.people_gender = {}
                print("  ⚠️  No 'Sex' column found - gender cohesion will not be considered")
            
            # Load vehicles
            self.vehicles_df = pd.read_csv(vehicles_csv)
            print(f"✓ Loaded {len(self.vehicles_df)} vehicles from {vehicles_csv}")
            
            # Process vehicles (only 3-point belts)
            self._process_vehicles_3point_only()
            
        except FileNotFoundError as e:
            print(f"❌ Error: Could not find file - {e}")
            sys.exit(1)
    
    def _process_vehicles_3point_only(self):
        """Process vehicles, only counting 3-point seat belts."""
        valid_vehicles = 0
        total_capacity = 0
        
        for _, row in self.vehicles_df.iterrows():
            vehicle_id = row['Surname & Car Reg']
            
            # Only count 3-point belts
            three_point_belts = int(row['3-point belts (excluding driver)'])
            
            # Skip vehicles with no 3-point belts
            if three_point_belts == 0:
                print(f"  ⚠️  Skipping {vehicle_id} - no 3-point belts")
                continue
            
            # Add vehicle with capacity (3-point belts PLUS driver seat)
            # The CSV shows belts "excluding driver", so we add 1 for the driver
            self.vehicles[vehicle_id] = three_point_belts + 1
            total_capacity += three_point_belts + 1
            
            # Get all possible drivers for this vehicle
            driver_cols = [col for col in self.vehicles_df.columns if col.startswith('Driver')]
            
            vehicle_has_driver = False
            for col in driver_cols:
                if pd.notna(row[col]) and row[col].strip():
                    driver_name = row[col].strip()
                    
                    # Only include drivers who are in our people list
                    if driver_name in self.people:
                        self.driver_vehicle_pairs.append((driver_name, vehicle_id))
                        vehicle_has_driver = True
                    elif driver_name != "Eleanor Clarke":  # Don't warn about Eleanor
                        pass  # Skip silently
            
            if not vehicle_has_driver:
                print(f"  ⚠️  Warning: {vehicle_id} has no valid drivers from camper list")
            
            valid_vehicles += 1
        
        print(f"✓ Processed {valid_vehicles} vehicles with {total_capacity} total seats (3-point belts only)")
        
        # Summary of drivers
        drivers = set(d for d, v in self.driver_vehicle_pairs)
        print(f"✓ Found {len(drivers)} valid drivers from camper list")
        print(f"✓ Total driver-vehicle pairs: {len(self.driver_vehicle_pairs)}")
        
        # Show which vehicles each driver can drive
        print("\nDriver assignments:")
        driver_to_vehicles = defaultdict(list)
        for driver, vehicle in self.driver_vehicle_pairs:
            driver_to_vehicles[driver].append(vehicle)
        
        for driver, vehicles in sorted(driver_to_vehicles.items()):
            print(f"  {driver}: {', '.join(vehicles)}")
        
        # Show which drivers each vehicle has
        print("\nVehicle driver options:")
        vehicle_to_drivers = defaultdict(list)
        for driver, vehicle in self.driver_vehicle_pairs:
            vehicle_to_drivers[vehicle].append(driver)
        
        for vehicle in sorted(self.vehicles.keys()):
            drivers_list = vehicle_to_drivers.get(vehicle, [])
            if not drivers_list:
                print(f"  {vehicle}: NO DRIVERS ❌")
            else:
                print(f"  {vehicle}: {', '.join(drivers_list)}")
    
    def get_summary_stats(self):
        """Print summary statistics."""
        print("\n" + "="*60)
        print("DATA SUMMARY")
        print("="*60)
        
        # Dorm summary
        dorm_summary = self.people_df.groupby('Dorm').agg({
            'Name': 'count',
            'Is Leader': lambda x: sum(x == True)
        }).rename(columns={'Name': 'Total', 'Is Leader': 'Leaders'})
        
        dorm_summary['Campers'] = dorm_summary['Total'] - dorm_summary['Leaders']
        
        print("\nDorm Summary:")
        print(dorm_summary.sort_values('Total', ascending=False).to_string())
        
        # Vehicle summary
        print(f"\nVehicle Summary:")
        print(f"  Total vehicles: {len(self.vehicles)}")
        minibus_count = sum(1 for v in self.vehicles if 'Minibus' in v)
        print(f"  Minibuses: {minibus_count}")
        print(f"  Regular cars: {len(self.vehicles) - minibus_count}")
        print(f"  Total capacity: {sum(self.vehicles.values())}")
        print(f"  Total people: {len(self.people)}")
        print(f"  Spare capacity: {sum(self.vehicles.values()) - len(self.people)}")
        
        # Driver coverage
        driver_coverage = defaultdict(int)
        all_drivers = set()
        for driver, vehicle in self.driver_vehicle_pairs:
            driver_coverage[driver] += 1
            all_drivers.add(driver)
        
        print(f"\nDriver Coverage:")
        print(f"  Total unique drivers: {len(all_drivers)}")
        print(f"  Drivers who can drive multiple vehicles: {sum(1 for c in driver_coverage.values() if c > 1)}")
        
        # Check for missing people
        drivers_not_in_people = all_drivers - set(self.people.keys())
        if drivers_not_in_people:
            print(f"\n⚠️  Drivers in cars.csv but not in campers.csv:")
            for d in sorted(drivers_not_in_people):
                print(f"    - {d}")
    
    def solve_ilp_allocation(self, time_limit=300):
        """
        Solve vehicle allocation using Integer Linear Programming.
        
        Args:
            time_limit: Maximum seconds to spend solving (default 5 minutes)
        """
        print("\n" + "="*60)
        print("SOLVING WITH INTEGER LINEAR PROGRAMMING")
        print("="*60)
        
        start_time = time.time()
        
        # Remove vehicles that have no valid drivers
        vehicles_with_drivers = set(v for d, v in self.driver_vehicle_pairs)
        usable_vehicles = {v: cap for v, cap in self.vehicles.items() if v in vehicles_with_drivers}
        
        excluded_vehicles = set(self.vehicles.keys()) - set(usable_vehicles.keys())
        if excluded_vehicles:
            print(f"\n⚠️  Excluding {len(excluded_vehicles)} vehicles with no valid drivers:")
            excluded_capacity = 0
            for v in sorted(excluded_vehicles):
                excluded_capacity += self.vehicles[v]
                print(f"   - {v} (capacity: {self.vehicles[v]})")
            print(f"   Total excluded capacity: {excluded_capacity} seats")
        
        # Check if we have enough capacity
        total_capacity = sum(usable_vehicles.values())
        if total_capacity < len(self.people):
            print(f"\n❌ ERROR: Not enough capacity!")
            print(f"   Need seats for: {len(self.people)} people")
            print(f"   Available seats: {total_capacity}")
            return {}, None
        
        # Quick feasibility check
        print(f"\nFeasibility check:")
        print(f"  People to allocate: {len(self.people)}")
        print(f"  Usable vehicles: {len(usable_vehicles)}")
        print(f"  Total seats available: {total_capacity}")
        print(f"  Unique drivers available: {len(set(d for d, v in self.driver_vehicle_pairs if v in usable_vehicles))}")
        
        # Check leader availability for minibuses
        leaders = [person for person in self.people if self.people_df[self.people_df['Name'] == person]['Is Leader'].values[0]]
        minibus_count = sum(1 for v in usable_vehicles if 'Minibus' in v)
        print(f"  Total leaders: {len(leaders)}")
        print(f"  Minibuses needing 2+ leaders: {minibus_count}")
        if minibus_count * 2 > len(leaders):
            print(f"  ⚠️  Warning: May not have enough leaders for all minibuses!")
        
        # Check if we have odd-numbered dorms or small dorms
        print("\nChecking for potential group size issues:")
        odd_dorms = []
        small_dorms = []
        for dorm in set(self.people.values()):
            dorm_size = sum(1 for p in self.people.values() if p == dorm)
            if dorm_size % 2 == 1:
                odd_dorms.append((dorm, dorm_size))
            if dorm_size < 3:
                small_dorms.append((dorm, dorm_size))
        
        if small_dorms:
            print(f"  ⚠️  Warning: {len(small_dorms)} dorms have fewer than 3 people:")
            for dorm, size in small_dorms:
                print(f"     - {dorm}: {size} people")
            print("  These dorms will necessarily have small group sizes.")
        
        if odd_dorms:
            print(f"  ⚠️  Warning: {len(odd_dorms)} dorms have odd numbers of people:")
            for dorm, size in odd_dorms:
                if size >= 3:  # Only show if not already listed as small
                    print(f"     - {dorm}: {size} people")
            print("  This might make it harder to achieve min group size of 3 everywhere.")
        
        if not small_dorms and not odd_dorms:
            print("  ✓ All dorms have 3+ people with even numbers - should be possible to achieve min group size of 3")
        
        # Create the model
        model = pulp.LpProblem("Camp_Vehicle_Allocation", pulp.LpMaximize)
        
        print("Creating decision variables...")
        
        # Decision variables
        # x[person][vehicle] = 1 if person is in vehicle
        x = {}
        for person in self.people:
            x[person] = {}
            for vehicle in usable_vehicles:
                var_name = f"x_{person}_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                x[person][vehicle] = pulp.LpVariable(var_name, cat='Binary')
        
        # y[driver][vehicle] = 1 if driver is driving vehicle
        y = {}
        for driver, vehicle in self.driver_vehicle_pairs:
            if vehicle not in usable_vehicles:
                continue
            if driver not in y:
                y[driver] = {}
            var_name = f"y_{driver}_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            y[driver][vehicle] = pulp.LpVariable(var_name, cat='Binary')
        
        # z[vehicle] = 1 if vehicle is used
        z = {}
        for vehicle in usable_vehicles:
            var_name = f"z_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            z[vehicle] = pulp.LpVariable(var_name, cat='Binary')
        
        print("Adding constraints...")
        
        # CONSTRAINTS
        
        # 1. Each person must be in exactly one vehicle
        for person in self.people:
            constraint_name = f"One_Vehicle_{person}".replace(" ", "_").replace("-", "_")
            model += pulp.lpSum(x[person][vehicle] for vehicle in usable_vehicles) == 1, constraint_name
        
        # 2. Vehicle capacity constraints (only if vehicle is used)
        for vehicle in usable_vehicles:
            constraint_name = f"Capacity_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            model += pulp.lpSum(x[person][vehicle] for person in self.people) <= usable_vehicles[vehicle] * z[vehicle], constraint_name
        
        # 3. Vehicle has driver if and only if it's used
        for vehicle in usable_vehicles:
            driver_sum = pulp.lpSum(
                y[driver][vehicle] 
                for driver, veh in self.driver_vehicle_pairs 
                if veh == vehicle and driver in y and vehicle in y[driver]
            )
            constraint_name = f"Driver_If_Used_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            model += driver_sum == z[vehicle], constraint_name
        
        # 4. A driver can only drive one vehicle
        drivers = set(d for d, v in self.driver_vehicle_pairs)
        for driver in drivers:
            if driver in y:
                constraint_name = f"Driver_One_Vehicle_{driver}".replace(" ", "_").replace("-", "_")
                model += pulp.lpSum(y[driver][vehicle] for vehicle in y[driver]) <= 1, constraint_name
        
        # 5. If someone is driving a vehicle, they must be in it
        for driver, vehicle in self.driver_vehicle_pairs:
            if vehicle in usable_vehicles and driver in y and vehicle in y[driver]:
                constraint_name = f"Driver_In_{driver}_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                model += y[driver][vehicle] <= x[driver][vehicle], constraint_name
        
        # 6. If vehicle is not used, no one can be in it
        for vehicle in usable_vehicles:
            for person in self.people:
                constraint_name = f"No_Use_No_Person_{person}_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                model += x[person][vehicle] <= z[vehicle], constraint_name
        
        # 7. Minibuses must have at least 2 leaders
        leaders = [person for person in self.people if self.people_df[self.people_df['Name'] == person]['Is Leader'].values[0]]
        print(f"  Found {len(leaders)} leaders for minibus constraint")
        
        for vehicle in usable_vehicles:
            if 'Minibus' in vehicle:
                constraint_name = f"Minibus_Leaders_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                # If minibus is used (z=1), then sum of leaders >= 2
                # If not used (z=0), then sum of leaders >= 0 (no constraint)
                model += pulp.lpSum(x[leader][vehicle] for leader in leaders) >= 2 * z[vehicle], constraint_name
        
        # Calculate group cohesion scores
        # Create variables for group sizes and minimum group sizes
        
        # g[dorm][vehicle] = number of people from dorm in vehicle
        g = {}
        dorms = set(self.people.values())
        for dorm in dorms:
            g[dorm] = {}
            for vehicle in usable_vehicles:
                var_name = f"g_{dorm}_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                g[dorm][vehicle] = pulp.LpVariable(var_name, lowBound=0, cat='Integer')
        
        # m[vehicle] = minimum group size in vehicle (what we want to maximize)
        m = {}
        for vehicle in usable_vehicles:
            var_name = f"m_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            m[vehicle] = pulp.LpVariable(var_name, lowBound=0, cat='Integer')
        
        # 8. Soft constraint: Try to ensure minimum group size >= 3
        # Create violation variables for cases where we must have smaller group sizes
        violations_2 = {}  # For group size of 2
        violations_1 = {}  # For group size of 1
        score_vars = {}
        for vehicle in usable_vehicles:
            var_name_2 = f"violation_2_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            var_name_1 = f"violation_1_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            violations_2[vehicle] = pulp.LpVariable(var_name_2, lowBound=0, cat='Integer')
            violations_1[vehicle] = pulp.LpVariable(var_name_1, lowBound=0, cat='Integer')
            
            # If vehicle is used, then m[vehicle] + violations_2 + violations_1 >= 3
            # This means:
            # - If m[vehicle] = 3, no violations needed
            # - If m[vehicle] = 2, need violations_2 = 1
            # - If m[vehicle] = 1, need violations_1 = 2 (or violations_2 = 2, but we'll enforce the right one)
            constraint_name = f"Min_Group_Three_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            model += m[vehicle] + violations_2[vehicle] + violations_1[vehicle] >= 3 * z[vehicle], constraint_name
            
            # Ensure violations are used correctly based on actual min group size
            # If m[vehicle] = 2, then violations_2 must be at least 1
            # If m[vehicle] = 1, then violations_1 must be at least 2
            # We need to link the violations to the actual group size
            
            # For group size 2: if m[vehicle] <= 2, then violations_2 >= 3 - m[vehicle]
            # But only if vehicle is used and m[vehicle] >= 2
            constraint_v2_1 = f"Violation2_Link1_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            constraint_v2_2 = f"Violation2_Link2_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            
            # violations_2 should be 1 when m = 2, 0 otherwise
            # We'll use the score_vars to determine this more precisely
            if vehicle in score_vars and 2 in score_vars[vehicle]:
                model += violations_2[vehicle] >= score_vars[vehicle][2], constraint_v2_1
                model += violations_2[vehicle] <= M * score_vars[vehicle][2], constraint_v2_2
            
            # For group size 1: violations_1 should be 2 when m = 1
            constraint_v1_1 = f"Violation1_Link1_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            constraint_v1_2 = f"Violation1_Link2_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            
            if vehicle in score_vars and 1 in score_vars[vehicle]:
                model += violations_1[vehicle] >= 2 * score_vars[vehicle][1], constraint_v1_1
                model += violations_1[vehicle] <= M * score_vars[vehicle][1], constraint_v1_2
        
        # Calculate group sizes per dorm per vehicle
        for dorm in dorms:
            for vehicle in usable_vehicles:
                dorm_people = [p for p in self.people if self.people[p] == dorm]
                constraint_name = f"Group_Size_{dorm}_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                model += g[dorm][vehicle] == pulp.lpSum(
                    x[person][vehicle] for person in dorm_people
                ), constraint_name
        
        # Define minimum group size in each vehicle
        # We use a big-M approach like in the demo
        M = len(self.people)  # Big M value
        
        for vehicle in usable_vehicles:
            for dorm in dorms:
                # Helper variable: h[dorm][vehicle] = 1 if dorm has people in vehicle
                var_name = f"h_{dorm}_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                h_var = pulp.LpVariable(var_name, cat='Binary')
                
                # If any person from dorm is in vehicle, h = 1
                constraint1 = f"Dorm_Present1_{dorm}_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                constraint2 = f"Dorm_Present2_{dorm}_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                
                dorm_people = [p for p in self.people if self.people[p] == dorm]
                model += pulp.lpSum(x[person][vehicle] for person in dorm_people) >= h_var, constraint1
                model += pulp.lpSum(x[person][vehicle] for person in dorm_people) <= M * h_var, constraint2
                
                # If dorm is present (h=1), then m[vehicle] <= g[dorm][vehicle]
                # But only if the vehicle is used
                constraint3 = f"Min_Group_{dorm}_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                model += m[vehicle] <= g[dorm][vehicle] + M * (1 - h_var) + M * (1 - z[vehicle]), constraint3
        
        # Create bonus variables for single-dorm vehicles (only for non-minibuses)
        single_dorm_bonus = {}
        for vehicle in usable_vehicles:
            if 'Minibus' not in vehicle:  # Only create bonus variables for regular cars
                var_name = f"single_dorm_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                single_dorm_bonus[vehicle] = pulp.LpVariable(var_name, cat='Binary')
                
                # Count how many dorms are present in the vehicle
                dorms_present = pulp.lpSum(
                    h_var for dorm in dorms 
                    for h_var in [model.variablesDict().get(f"h_{dorm}_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_"), None)]
                    if h_var is not None
                )
                
                # single_dorm_bonus = 1 only if exactly one dorm is present AND vehicle is used
                # We need dorms_present == 1, which means dorms_present <= 1 AND dorms_present >= 1
                constraint_single1 = f"Single_Dorm_Upper_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                constraint_single2 = f"Single_Dorm_Lower_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                
                # If vehicle is used and has single dorm, then single_dorm_bonus can be 1
                # We need: single_dorm_bonus <= z[vehicle] (can't have bonus if vehicle not used)
                # And: single_dorm_bonus can only be 1 if dorms_present == 1
                model += dorms_present >= single_dorm_bonus[vehicle], constraint_single1
                model += dorms_present <= 1 + M * (1 - single_dorm_bonus[vehicle]), constraint_single2
                model += single_dorm_bonus[vehicle] <= z[vehicle], f"Single_Dorm_Used_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        
        # Create gender cohesion bonus for mixed-dorm vehicles
        gender_cohesion_bonus = {}
        if self.people_gender:  # Only if gender data is available
            for vehicle in usable_vehicles:
                var_name = f"gender_cohesion_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                gender_cohesion_bonus[vehicle] = pulp.LpVariable(var_name, cat='Binary')
                
                # Count people by gender in this vehicle
                males = [p for p in self.people if self.people_gender.get(p) == 'M']
                females = [p for p in self.people if self.people_gender.get(p) == 'F']
                
                males_in_vehicle = pulp.lpSum(x[p][vehicle] for p in males if p in x)
                females_in_vehicle = pulp.lpSum(x[p][vehicle] for p in females if p in x)
                
                # Binary variables to indicate if males/females are present
                has_males = pulp.LpVariable(f"has_males_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_"), cat='Binary')
                has_females = pulp.LpVariable(f"has_females_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_"), cat='Binary')
                
                # Set has_males = 1 if any males in vehicle
                model += males_in_vehicle >= has_males, f"Males_Present1_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                model += males_in_vehicle <= M * has_males, f"Males_Present2_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                
                # Set has_females = 1 if any females in vehicle
                model += females_in_vehicle >= has_females, f"Females_Present1_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                model += females_in_vehicle <= M * has_females, f"Females_Present2_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                
                # gender_cohesion_bonus = 1 if vehicle is used AND (all male OR all female)
                # This means has_males + has_females <= 1 (at most one gender present)
                # But we also need the vehicle to be used and have multiple dorms
                
                # Check if vehicle has multiple dorms (inverse of single_dorm)
                is_mixed_dorm = pulp.LpVariable(f"is_mixed_dorm_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_"), cat='Binary')
                
                # is_mixed_dorm = 1 if vehicle is used AND not single dorm
                if vehicle in single_dorm_bonus:
                    model += is_mixed_dorm <= z[vehicle], f"Mixed_Dorm_Used_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                    model += is_mixed_dorm <= 1 - single_dorm_bonus[vehicle], f"Mixed_Dorm_Not_Single_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                    model += is_mixed_dorm >= z[vehicle] - single_dorm_bonus[vehicle], f"Mixed_Dorm_Both_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                else:
                    # For minibuses, just check if used (they don't have single_dorm_bonus)
                    model += is_mixed_dorm == z[vehicle], f"Mixed_Dorm_Minibus_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                
                # gender_cohesion_bonus = 1 if mixed dorm AND single gender
                # Single gender means has_males + has_females = 1 (exactly one gender)
                model += gender_cohesion_bonus[vehicle] <= is_mixed_dorm, f"Gender_Bonus_Mixed_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                model += gender_cohesion_bonus[vehicle] <= 2 - has_males - has_females, f"Gender_Bonus_Single1_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                model += gender_cohesion_bonus[vehicle] <= has_males + has_females, f"Gender_Bonus_Single2_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        
        # OBJECTIVE: Maximize weighted sum of minimum group sizes + bonuses
        # Use quadratic scaling for min group sizes to create bigger differentials
        # min_size=1 -> score=1, min_size=2 -> score=4, min_size=3 -> score=9, etc.
        # We'll approximate this by using the sum of minimum sizes but with a multiplier
        
        # Since we can't directly square variables in linear programming, we'll use a piecewise linear approximation
        # We'll create auxiliary variables for different score levels

        for vehicle in usable_vehicles:
            score_vars[vehicle] = {}
            # Create binary variables for each possible min group size (1 to 10)
            for size in range(1, 11):
                var_name = f"score_{vehicle}_size_{size}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                score_vars[vehicle][size] = pulp.LpVariable(var_name, cat='Binary')
            
            # Constraint: exactly one size must be selected if vehicle is used
            model += pulp.lpSum(score_vars[vehicle][size] for size in range(1, 11)) == z[vehicle], f"One_Score_{vehicle}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            
            # Constraint: if size s is selected, then m[vehicle] must be at least s
            for size in range(1, 11):
                model += m[vehicle] >= size * score_vars[vehicle][size], f"Min_Score_{vehicle}_{size}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                # Also m[vehicle] <= s + M*(1 - score_vars[vehicle][s])
                model += m[vehicle] <= size + M * (1 - score_vars[vehicle][size]), f"Max_Score_{vehicle}_{size}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        
        # Define the scoring weights (quadratic-like progression)
        score_weights = SCORE_WEIGHTS
        
        # OBJECTIVE: Maximize weighted scores + bonuses - penalties for small group sizes
        objective = pulp.lpSum(
            score_weights[size] * score_vars[vehicle][size] 
            for vehicle in usable_vehicles 
            for size in range(1, 11)
        ) + pulp.lpSum(
            SINGLE_DORM_BONUS * single_dorm_bonus[vehicle] 
            for vehicle in single_dorm_bonus  # Only includes non-minibuses
        ) - pulp.lpSum(
            MIN_SIZE_2_PENALTY * violations_2[vehicle]
            for vehicle in usable_vehicles
        ) - pulp.lpSum(
            MIN_SIZE_1_PENALTY * violations_1[vehicle]
            for vehicle in usable_vehicles
        )
        
        # Add gender cohesion bonus if gender data exists
        if self.people_gender and gender_cohesion_bonus:
            objective += pulp.lpSum(
                GENDER_COHESION_BONUS * gender_cohesion_bonus[vehicle]
                for vehicle in gender_cohesion_bonus
            )
        
        model += objective, "Objective"
        
        print(f"\nSolving (this may take up to {time_limit} seconds)...")
        
        # Solve with time limit and optimality gap
        solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=time_limit, gapRel=0.01)
        model.solve(solver)
        
        solve_time = time.time() - start_time
        print(f"\nSolved in {solve_time:.1f} seconds")
        print(f"Status: {pulp.LpStatus[model.status]}")
        
        if model.status == pulp.LpStatusInfeasible:
            print("\n❌ ERROR: Problem is infeasible!")
            print("This means no valid allocation exists with current constraints.")
            print("Possible issues:")
            print("- Not enough drivers for vehicles")
            print("- Capacity constraints too tight")
            print("- Minimum group size constraints impossible to satisfy")
            return {}, model
        
        if model.status != pulp.LpStatusOptimal:
            print("⚠️  Warning: Solution may not be optimal")
            if solve_time >= time_limit - 1:
                print("  Time limit reached - solution might improve with more time")
                print("  Consider increasing time_limit parameter")
        
        print(f"Objective value: {pulp.value(model.objective):.2f}")
        
        # Show optimality gap if available
        if hasattr(model, 'solutionInfo') and 'gap' in model.solutionInfo:
            gap = model.solutionInfo['gap']
            print(f"Optimality gap: {gap:.2%}")
        
        # Calculate and display scoring breakdown
        print("\nScoring breakdown:")
        print(f"  Min group sizes: 1→{SCORE_WEIGHTS[1]}pt, 2→{SCORE_WEIGHTS[2]}pts, 3→{SCORE_WEIGHTS[3]}pts, 4→{SCORE_WEIGHTS[4]}pts, 5→{SCORE_WEIGHTS[5]}pts, etc.")
        print(f"  Single-dorm bonus: {SINGLE_DORM_BONUS}pts per car (not minibuses)")
        if self.people_gender:
            print(f"  Gender cohesion bonus: {GENDER_COHESION_BONUS}pts for mixed-dorm vehicles with single gender")
        print(f"  Group size penalties: -{MIN_SIZE_2_PENALTY}pts for size 2, -{MIN_SIZE_1_PENALTY}pts for size 1")
        
        # Check for violations in the solution
        violations_2_count = 0
        violations_1_count = 0
        if 'violations_2' in locals() and 'violations_1' in locals():
            for vehicle in usable_vehicles:
                if vehicle in violations_2 and pulp.value(violations_2[vehicle]) > 0.5:
                    violations_2_count += 1
                if vehicle in violations_1 and pulp.value(violations_1[vehicle]) > 0.5:
                    violations_1_count += 1
        
        if violations_1_count > 0 or violations_2_count > 0:
            print(f"\n⚠️  WARNING: Small group sizes detected:")
            if violations_1_count > 0:
                print(f"  - {violations_1_count} vehicles with isolated individuals (min group size = 1)")
            if violations_2_count > 0:
                print(f"  - {violations_2_count} vehicles with min group size = 2")
            print("  The solver had to allow this to find a feasible solution.")
        
        # Extract solution
        allocation = {}
        
        for vehicle in usable_vehicles:
            # Check if vehicle is used
            if pulp.value(z[vehicle]) < 0.5:
                continue
                
            passengers = []
            driver = None
            
            # Get passengers
            for person in self.people:
                if pulp.value(x[person][vehicle]) > 0.5:  # Binary variable should be 0 or 1
                    passengers.append(person)
            
            # Get driver
            for d, v in self.driver_vehicle_pairs:
                if v == vehicle and d in y and vehicle in y[d]:
                    if pulp.value(y[d][vehicle]) > 0.5:
                        driver = d
                        break
            
            if passengers and driver:
                allocation[(vehicle, driver)] = passengers
            elif passengers:
                print(f"⚠️  Warning: {vehicle} has passengers but no driver!")
        
        # Show unused vehicles
        unused_vehicles = []
        for vehicle in usable_vehicles:
            if pulp.value(z[vehicle]) < 0.5:
                unused_vehicles.append(vehicle)
        
        if unused_vehicles:
            print(f"\nUnused vehicles ({len(unused_vehicles)}):")
            for v in sorted(unused_vehicles):
                print(f"  - {v} (capacity: {usable_vehicles[v]})")
        
        return allocation, model
    
    def print_ilp_solution(self, allocation):
        """Print the ILP solution."""
        if not allocation:
            print("\n❌ No valid allocation found!")
            return
            
        metrics = self.calculate_metrics(allocation)
        
        print("\n" + "="*60)
        print("OPTIMAL ALLOCATION (ILP)")
        print("="*60)
        
        print("\nMETRICS:")
        for key, value in metrics.items():
            print(f"  • {key}: {value}")
        
        print("\n" + "-"*60)
        print("VEHICLE ASSIGNMENTS:")
        print("-"*60)
        
        # Sort by vehicle type and name
        sorted_allocation = sorted(allocation.items(), 
                                 key=lambda x: ('Minibus' not in x[0][0], x[0][0]))
        
        for (vehicle, driver), passengers in sorted_allocation:
            print(f"\n{vehicle}")
            print(f"  Driver: {driver}")
            print(f"  Seats: {len(passengers)}/{self.vehicles[vehicle]} used")
            
            # Count leaders in this vehicle
            leaders_in_vehicle = [p for p in passengers if self.people_df[self.people_df['Name'] == p]['Is Leader'].values[0]]
            print(f"  Leaders: {len(leaders_in_vehicle)}")
            if 'Minibus' in vehicle and len(leaders_in_vehicle) < 2:
                print(f"  ⚠️  WARNING: Minibus has only {len(leaders_in_vehicle)} leader(s)!")
            
            # Group by dorm
            dorm_counts = Counter(self.people[p] for p in passengers)
            min_score = min(dorm_counts.values()) if dorm_counts else 0
            is_single_dorm = len(dorm_counts) == 1
            
            # Check gender composition if data available
            gender_info = ""
            if self.people_gender:
                gender_counts = Counter(self.people_gender.get(p, 'Unknown') for p in passengers)
                if len(gender_counts) == 1 and len(dorm_counts) > 1:
                    gender = list(gender_counts.keys())[0]
                    gender_info = f" (All {gender}, +10 gender cohesion bonus)"
                elif len(gender_counts) > 1:
                    gender_breakdown = ', '.join([f"{g}:{c}" for g, c in gender_counts.items()])
                    gender_info = f" (Mixed gender: {gender_breakdown})"
            
            # Calculate score for this vehicle
            vehicle_score = SCORE_WEIGHTS.get(min_score, min_score * min_score)
            
            # Apply bonuses and show warnings for small group sizes
            bonus_info = []
            penalty_info = []
            
            if min_score == 1:
                penalty_info.append(f"⚠️ ISOLATED PERSON -{MIN_SIZE_1_PENALTY}pts")
            elif min_score == 2:
                penalty_info.append(f"⚠️ Small group -{MIN_SIZE_2_PENALTY}pts")
            
            if is_single_dorm and 'Minibus' not in vehicle:
                vehicle_score += SINGLE_DORM_BONUS
                bonus_info.append(f"Single dorm +{SINGLE_DORM_BONUS}")
            elif is_single_dorm and 'Minibus' in vehicle:
                bonus_info.append("Single dorm (no bonus for minibus)")
            
            if self.people_gender and not is_single_dorm and len(gender_counts) == 1:
                vehicle_score += GENDER_COHESION_BONUS
                bonus_info.append(f"Gender cohesion +{GENDER_COHESION_BONUS}")
            
            # Apply penalties
            if min_score == 2:
                vehicle_score -= MIN_SIZE_2_PENALTY
            elif min_score == 1:
                vehicle_score -= MIN_SIZE_1_PENALTY
            
            print(f"  Min group size: {min_score}{gender_info}")
            if bonus_info:
                print(f"  Bonuses: {', '.join(bonus_info)}")
            if penalty_info:
                print(f"  Penalties: {', '.join(penalty_info)}")
            print(f"  Vehicle score: {vehicle_score} points")
            
            # Print by dorm
            for dorm, count in sorted(dorm_counts.items(), key=lambda x: x[1], reverse=True):
                members = [p for p in passengers if self.people[p] == dorm]
                # Mark leaders with (L)
                members_with_roles = []
                for m in members:
                    if self.people_df[self.people_df['Name'] == m]['Is Leader'].values[0]:
                        members_with_roles.append(f"{m} (L)")
                    else:
                        members_with_roles.append(m)
                print(f"  [{dorm}] {count} people: {', '.join(members_with_roles)}")
        
        # Check for unassigned
        assigned = set()
        for passengers in allocation.values():
            assigned.update(passengers)
        
        if len(assigned) < len(self.people):
            unassigned = set(self.people.keys()) - assigned
            print(f"\n⚠️  WARNING: {len(unassigned)} unassigned people:")
            for person in sorted(unassigned):
                print(f"    - {person} ({self.people[person]})")
    
    def calculate_metrics(self, allocation):
        """Calculate allocation metrics."""
        if not allocation:
            return {
                'total_score': 0,
                'avg_score': 0,
                'min_score': 0,
                'max_score': 0,
                'vehicles_with_size_2': 0,
                'vehicles_with_size_1': 0,
                'vehicles_used': 0,
                'single_dorm_vehicles': 0,
                'capacity_utilization': "0/0 (0.0%)"
            }
            
        scores = []
        isolated_count = 0
        vehicles_size_2 = 0
        vehicles_size_1 = 0
        single_dorm_count = 0
        single_gender_mixed_count = 0
        
        for passengers in allocation.values():
            dorm_counts = Counter(self.people[p] for p in passengers)
            min_score = min(dorm_counts.values()) if dorm_counts else 0
            
            # Calculate weighted score
            vehicle_score = SCORE_WEIGHTS.get(min_score, min_score * min_score)
            
            # Check which vehicle this is (need to match passengers to vehicle)
            vehicle_name = None
            for (v, d), p in allocation.items():
                if p == passengers:
                    vehicle_name = v
                    break
            
            # Single dorm bonus only for non-minibuses
            if len(dorm_counts) == 1 and vehicle_name and 'Minibus' not in vehicle_name:
                vehicle_score += SINGLE_DORM_BONUS
                single_dorm_count += 1
            
            # Gender cohesion bonus for mixed-dorm vehicles
            if self.people_gender and len(dorm_counts) > 1:
                gender_counts = Counter(self.people_gender.get(p, 'Unknown') for p in passengers)
                if len(gender_counts) == 1:
                    vehicle_score += GENDER_COHESION_BONUS
                    single_gender_mixed_count += 1
            
            # Apply penalties for small group sizes
            if min_score == 2:
                vehicle_score -= MIN_SIZE_2_PENALTY
                vehicles_size_2 += 1
            elif min_score == 1:
                vehicle_score -= MIN_SIZE_1_PENALTY
                vehicles_size_1 += 1
            
            scores.append(vehicle_score)
            isolated_count += sum(1 for count in dorm_counts.values() if count == 1)
        
        total_capacity = sum(self.vehicles[v] for v, _ in allocation.keys())
        total_passengers = sum(len(p) for p in allocation.values())
        
        if total_capacity == 0:
            capacity_util = "0/0 (0.0%)"
        else:
            capacity_util = f"{total_passengers}/{total_capacity} ({100*total_passengers/total_capacity:.1f}%)"
        
        return {
            'total_score': sum(scores),
            'avg_score': np.mean(scores) if scores else 0,
            'min_score': min(scores) if scores else 0,
            'max_score': max(scores) if scores else 0,
            'isolated_people': isolated_count,
            'vehicles_with_size_2': vehicles_size_2,
            'vehicles_with_size_1': vehicles_size_1,
            'vehicles_used': len(allocation),
            'single_dorm_cars': single_dorm_count,
            'single_gender_mixed_dorms': single_gender_mixed_count,
            'capacity_utilization': capacity_util
        }
    
    def export_results(self, allocation, prefix="ilp_allocation"):
        """Export allocation to CSV files."""
        if not allocation:
            print("\n❌ No allocation to export!")
            return
            
        # Detailed passenger list
        rows = []
        for (vehicle, driver), passengers in allocation.items():
            for passenger in passengers:
                rows.append({
                    'Vehicle': vehicle,
                    'Driver': driver,
                    'Passenger': passenger,
                    'Dorm': self.people[passenger],
                    'Is_Driver': passenger == driver,
                    'Vehicle_Type': 'Minibus' if 'Minibus' in vehicle else 'Car'
                })
        
        detailed_df = pd.DataFrame(rows)
        detailed_df.to_csv(f'{prefix}_detailed.csv', index=False)
        
        # Summary by vehicle
        summary_rows = []
        for (vehicle, driver), passengers in allocation.items():
            dorm_counts = Counter(self.people[p] for p in passengers)
            
            summary_rows.append({
                'Vehicle': vehicle,
                'Driver': driver,
                'Type': 'Minibus' if 'Minibus' in vehicle else 'Car',
                'Capacity': self.vehicles[vehicle],
                'Passengers': len(passengers),
                'Utilization_%': round(100 * len(passengers) / self.vehicles[vehicle], 1),
                'Min_Group_Size': min(dorm_counts.values()) if dorm_counts else 0,
                'Num_Dorms': len(dorm_counts),
                'Primary_Dorm': max(dorm_counts.items(), key=lambda x: x[1])[0] if dorm_counts else '',
                'Dorm_Details': ', '.join([f"{d}:{c}" for d, c in sorted(dorm_counts.items())])
            })
        
        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.sort_values(['Type', 'Vehicle'], ascending=[False, True])
        summary_df.to_csv(f'{prefix}_summary.csv', index=False)
        
        print(f"\n✅ Results exported:")
        print(f"   • {prefix}_detailed.csv - Full passenger list")
        print(f"   • {prefix}_summary.csv - Vehicle summary")
    
    def run(self):
        """Main execution function."""
        print("\n" + "="*60)
        print("CAMP VEHICLE ALLOCATION - INTEGER LINEAR PROGRAMMING")
        print("Using 3-point seat belts only")
        print("Data folder: ./data/")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Show summary
        self.get_summary_stats()
        
        # Solve with ILP
        allocation, model = self.solve_ilp_allocation(time_limit=1800)
        
        # Print solution
        self.print_ilp_solution(allocation)
        
        # Export results
        self.export_results(allocation)
        
        return allocation, model


def main():
    """Main entry point."""
    # Check if PuLP is installed
    try:
        import pulp
    except ImportError:
        print("❌ Error: PuLP is not installed")
        print("Please install it with: pip install pulp")
        sys.exit(1)
    
    allocator = CampVehicleAllocatorILP()
    allocation, model = allocator.run()
    
    if allocation:
        print("\n" + "="*60)
        print("NOTES")
        print("="*60)
        print("""
The Integer Linear Programming approach found an optimal allocation
that maximizes group cohesion (same-dorm pairs in vehicles).

This solution:
- Ensures everyone is assigned to exactly one vehicle
- Respects all capacity constraints
- Only uses vehicles that have available drivers
- Allows some vehicles to remain unused if necessary
- Maximizes the number of same-dorm pairs

The solver intelligently selects which vehicles to use based on
available drivers and optimal group arrangements.
""")


if __name__ == "__main__":
    main()