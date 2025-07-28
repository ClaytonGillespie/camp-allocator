#!/usr/bin/env python3
"""
Camp Vehicle Allocation System
Allocates campers to vehicles while keeping dorm groups together.
Only uses vehicles with 3-point seat belts.
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import random
import sys

class CampVehicleAllocator:
    def __init__(self):
        """Initialize the allocator."""
        self.people_df = None
        self.vehicles_df = None
        self.people_groups = {}
        self.all_people = set()
        self.vehicles = {}
        self.driver_vehicles = defaultdict(list)
        self.vehicle_drivers = defaultdict(list)
        
    def load_data(self, campers_csv='./data/campers.csv', vehicles_csv='./data/cars.csv'):
        """Load camper and vehicle data from CSV files."""
        try:
            # Load campers
            self.people_df = pd.read_csv(campers_csv)
            print(f"✓ Loaded {len(self.people_df)} people from {campers_csv}")
            
            # Load vehicles
            self.vehicles_df = pd.read_csv(vehicles_csv)
            print(f"✓ Loaded {len(self.vehicles_df)} vehicles from {vehicles_csv}")
            
            # Create people_groups mapping
            self.people_groups = dict(zip(self.people_df['Name'], self.people_df['Dorm']))
            self.all_people = set(self.people_df['Name'])
            
            # Process vehicles (only 3-point belts)
            self._process_vehicles_3point_only()
            
        except FileNotFoundError as e:
            print(f"❌ Error: Could not find file - {e}")
            sys.exit(1)
    
    def _process_vehicles_3point_only(self):
        """Process vehicles, only counting 3-point seat belts."""
        valid_vehicles = 0
        
        for idx, row in self.vehicles_df.iterrows():
            vehicle_id = row['Surname & Car Reg']
            
            # Only count 3-point belts
            three_point_belts = row['3-point belts (excluding driver)']
            
            # Skip vehicles with no 3-point belts
            if three_point_belts == 0:
                print(f"  ⚠️  Skipping {vehicle_id} - no 3-point belts")
                continue
            
            # Get all possible drivers
            driver_cols = [col for col in self.vehicles_df.columns if col.startswith('Driver')]
            drivers = []
            
            for col in driver_cols:
                if pd.notna(row[col]) and row[col].strip():
                    driver_name = row[col].strip()
                    # Skip Eleanor Clarke as she's not in our camper list
                    if driver_name == "Eleanor Clarke":
                        continue
                    drivers.append(driver_name)
                    self.driver_vehicles[driver_name].append(vehicle_id)
            
            self.vehicles[vehicle_id] = {
                'capacity': int(three_point_belts),  # Only 3-point belts
                'drivers': drivers,
                'is_minibus': 'Minibus' in vehicle_id
            }
            
            for driver in drivers:
                self.vehicle_drivers[vehicle_id].append(driver)
            
            valid_vehicles += 1
        
        print(f"✓ Processed {valid_vehicles} vehicles with 3-point belts")
    
    def get_dorm_summary(self) -> pd.DataFrame:
        """Get summary statistics for each dorm."""
        summary = self.people_df.groupby('Dorm').agg({
            'Name': 'count',
            'Is Leader': lambda x: sum(x == True)
        }).rename(columns={'Name': 'Total', 'Is Leader': 'Leaders'})
        
        summary['Campers'] = summary['Total'] - summary['Leaders']
        
        # Add driver count
        driver_counts = {}
        for dorm in summary.index:
            dorm_people = self.people_df[self.people_df['Dorm'] == dorm]['Name'].tolist()
            driver_count = sum(1 for person in dorm_people if person in self.driver_vehicles)
            driver_counts[dorm] = driver_count
        
        summary['Drivers'] = pd.Series(driver_counts)
        
        return summary.sort_values('Total', ascending=False)
    
    def score_vehicle(self, passengers: List[str]) -> int:
        """Score based on minimum count from any dorm group."""
        if not passengers:
            return 0
        
        dorm_counts = Counter(self.people_groups[p] for p in passengers)
        return min(dorm_counts.values()) if dorm_counts else 0
    
    def score_allocation(self, allocation: Dict[Tuple[str, str], List[str]]) -> Dict[str, float]:
        """Score entire allocation and return detailed metrics."""
        scores = []
        isolated_count = 0
        
        for (vehicle, driver), passengers in allocation.items():
            score = self.score_vehicle(passengers)
            scores.append(score)
            
            # Count isolated people
            dorm_counts = Counter(self.people_groups[p] for p in passengers)
            isolated_count += sum(1 for count in dorm_counts.values() if count == 1)
        
        return {
            'total_score': sum(scores),
            'avg_score': np.mean(scores) if scores else 0,
            'min_score': min(scores) if scores else 0,
            'max_score': max(scores) if scores else 0,
            'isolated_people': isolated_count,
            'vehicles_used': len(allocation)
        }
    
    def allocate_vehicles(self) -> Dict[Tuple[str, str], List[str]]:
        """Allocate people to vehicles using greedy algorithm."""
        # Get dorm statistics
        dorm_summary = self.get_dorm_summary()
        sorted_dorms = dorm_summary.sort_values('Total', ascending=False).index.tolist()
        
        # Initialize tracking
        allocation = {}
        unassigned = self.all_people.copy()
        used_vehicles = set()
        assigned_drivers = set()
        
        # Separate vehicles by type
        minibuses = [v for v, info in self.vehicles.items() if info['is_minibus']]
        regular_vehicles = [v for v, info in self.vehicles.items() if not info['is_minibus']]
        
        print("\nAllocation process:")
        
        # First pass: Assign large dorms to minibuses
        for dorm in sorted_dorms:
            dorm_members = self.people_df[self.people_df['Dorm'] == dorm]['Name'].tolist()
            dorm_size = len([m for m in dorm_members if m in unassigned])
            
            if dorm_size >= 10:
                for minibus in minibuses:
                    if minibus in used_vehicles:
                        continue
                    
                    # Find available drivers
                    available_drivers = [
                        d for d in self.vehicles[minibus]['drivers']
                        if d in unassigned and d not in assigned_drivers
                    ]
                    
                    if available_drivers:
                        # Prefer driver from same dorm
                        dorm_drivers = [d for d in available_drivers if self.people_groups.get(d) == dorm]
                        driver = dorm_drivers[0] if dorm_drivers else available_drivers[0]
                        
                        # Allocate
                        capacity = self.vehicles[minibus]['capacity']
                        vehicle_passengers = [driver]
                        unassigned.remove(driver)
                        assigned_drivers.add(driver)
                        
                        # Add dorm members
                        members_to_add = [m for m in dorm_members if m != driver and m in unassigned]
                        for member in members_to_add[:capacity]:
                            vehicle_passengers.append(member)
                            unassigned.remove(member)
                        
                        allocation[(minibus, driver)] = vehicle_passengers
                        used_vehicles.add(minibus)
                        print(f"  → {dorm}: {len(vehicle_passengers)} people → {minibus}")
                        break
        
        # Second pass: Regular vehicles for remaining groups
        for dorm in sorted_dorms:
            dorm_members = [m for m in self.people_df[self.people_df['Dorm'] == dorm]['Name'].tolist() 
                           if m in unassigned]
            
            while dorm_members:
                best_match = None
                best_score = -1
                
                for vehicle in regular_vehicles:
                    if vehicle in used_vehicles:
                        continue
                    
                    available_drivers = [
                        d for d in self.vehicles[vehicle]['drivers']
                        if d in unassigned and d not in assigned_drivers
                    ]
                    
                    for driver in available_drivers:
                        # Score this combination
                        is_same_dorm = self.people_groups.get(driver) == dorm
                        can_fit = min(len(dorm_members), self.vehicles[vehicle]['capacity'])
                        score = can_fit + (10 if is_same_dorm else 0)
                        
                        if score > best_score:
                            best_score = score
                            best_match = (vehicle, driver)
                
                if best_match is None:
                    break
                
                vehicle, driver = best_match
                capacity = self.vehicles[vehicle]['capacity']
                
                # Allocate
                vehicle_passengers = [driver]
                unassigned.remove(driver)
                assigned_drivers.add(driver)
                used_vehicles.add(vehicle)
                
                # Add members
                if self.people_groups.get(driver) == dorm:
                    dorm_members = [m for m in dorm_members if m != driver]
                
                for member in dorm_members[:capacity]:
                    vehicle_passengers.append(member)
                    unassigned.remove(member)
                
                allocation[(vehicle, driver)] = vehicle_passengers
                dorm_members = [m for m in dorm_members if m in unassigned]
                
                print(f"  → {dorm}: {len(vehicle_passengers)} people → {vehicle}")
        
        # Final pass: Handle remaining unassigned
        self._handle_remaining_unassigned(allocation, unassigned, used_vehicles, assigned_drivers)
        
        return allocation
    
    def _handle_remaining_unassigned(self, allocation, unassigned, used_vehicles, assigned_drivers):
        """Handle any remaining unassigned people."""
        initial_unassigned = len(unassigned)
        
        # First try to fit in existing vehicles
        for person in list(unassigned):
            best_vehicle = None
            best_score = -1
            
            for (vehicle, driver), passengers in allocation.items():
                if len(passengers) - 1 < self.vehicles[vehicle]['capacity']:
                    # Score if we add this person
                    temp_passengers = passengers + [person]
                    score = self.score_vehicle(temp_passengers)
                    
                    if score > best_score:
                        best_score = score
                        best_vehicle = (vehicle, driver)
            
            if best_vehicle:
                allocation[best_vehicle].append(person)
                unassigned.remove(person)
        
        # Then try unused vehicles
        if unassigned:
            unused_vehicles = [v for v in self.vehicles if v not in used_vehicles]
            
            for vehicle in unused_vehicles:
                if not unassigned:
                    break
                
                available_drivers = [
                    d for d in self.vehicles[vehicle]['drivers']
                    if d in unassigned and d not in assigned_drivers
                ]
                
                if available_drivers:
                    driver = available_drivers[0]
                    capacity = self.vehicles[vehicle]['capacity']
                    
                    vehicle_passengers = [driver]
                    unassigned.remove(driver)
                    assigned_drivers.add(driver)
                    
                    for person in list(unassigned)[:capacity]:
                        vehicle_passengers.append(person)
                        unassigned.remove(person)
                    
                    allocation[(vehicle, driver)] = vehicle_passengers
                    used_vehicles.add(vehicle)
                    print(f"  → Mixed group: {len(vehicle_passengers)} people → {vehicle}")
        
        if initial_unassigned > len(unassigned):
            print(f"  → Fitted {initial_unassigned - len(unassigned)} remaining people into existing vehicles")
    
    def optimize_allocation(self, initial_allocation: Dict, iterations: int = 3000) -> Dict:
        """Optimize using simulated annealing."""
        current = {k: v[:] for k, v in initial_allocation.items()}
        current_metrics = self.score_allocation(current)
        best = current.copy()
        best_metrics = current_metrics
        
        temperature = 1.0
        cooling_rate = 0.995
        improvements = 0
        
        for i in range(iterations):
            if len(current) < 2:
                break
            
            # Pick two vehicles
            vehicles = list(current.keys())
            v1, v2 = random.sample(vehicles, 2)
            
            if len(current[v1]) <= 1 or len(current[v2]) <= 1:
                continue
            
            # Pick passengers to swap (not drivers)
            p1_idx = random.randint(1, len(current[v1]) - 1)
            p2_idx = random.randint(1, len(current[v2]) - 1)
            
            # Swap
            current[v1][p1_idx], current[v2][p2_idx] = current[v2][p2_idx], current[v1][p1_idx]
            
            new_metrics = self.score_allocation(current)
            
            # Calculate improvement
            delta = (new_metrics['avg_score'] - current_metrics['avg_score']) * 100 + \
                    (current_metrics['isolated_people'] - new_metrics['isolated_people']) * 10
            
            # Accept or reject
            if delta > 0 or random.random() < np.exp(delta / temperature):
                current_metrics = new_metrics
                if new_metrics['avg_score'] > best_metrics['avg_score'] or \
                   (new_metrics['avg_score'] == best_metrics['avg_score'] and 
                    new_metrics['isolated_people'] < best_metrics['isolated_people']):
                    best = {k: v[:] for k, v in current.items()}
                    best_metrics = new_metrics
                    improvements += 1
            else:
                # Revert
                current[v1][p1_idx], current[v2][p2_idx] = current[v2][p2_idx], current[v1][p1_idx]
            
            temperature *= cooling_rate
            
            # Progress update
            if (i + 1) % 500 == 0:
                print(f"  → Iteration {i+1}/{iterations}: {improvements} improvements made")
        
        print(f"  → Optimization complete: {improvements} total improvements")
        return best
    
    def print_allocation(self, allocation: Dict):
        """Pretty print the allocation."""
        metrics = self.score_allocation(allocation)
        
        print("\n" + "="*60)
        print("VEHICLE ALLOCATION RESULTS")
        print("="*60)
        
        print("\nMETRICS:")
        print(f"  • Average score: {metrics['avg_score']:.2f}")
        print(f"  • Total score: {metrics['total_score']}")
        print(f"  • Isolated people: {metrics['isolated_people']}")
        print(f"  • Vehicles used: {metrics['vehicles_used']}")
        
        # Calculate totals
        total_capacity = sum(self.vehicles[v]['capacity'] + 1 for v, _ in allocation.keys())
        total_passengers = sum(len(p) for p in allocation.values())
        
        print(f"  • Capacity utilization: {total_passengers}/{total_capacity} ({100*total_passengers/total_capacity:.1f}%)")
        
        print("\n" + "-"*60)
        print("VEHICLE ASSIGNMENTS:")
        print("-"*60)
        
        # Sort by vehicle type and name
        sorted_allocation = sorted(allocation.items(), 
                                 key=lambda x: (not self.vehicles[x[0][0]]['is_minibus'], x[0][0]))
        
        for (vehicle, driver), passengers in sorted_allocation:
            vehicle_info = self.vehicles[vehicle]
            
            print(f"\n{vehicle}")
            print(f"  Driver: {driver}")
            print(f"  Seats: {len(passengers)}/{vehicle_info['capacity'] + 1} used")
            print(f"  Score: {self.score_vehicle(passengers)}")
            
            # Group by dorm
            dorm_groups = defaultdict(list)
            for p in passengers:
                dorm_groups[self.people_groups[p]].append(p)
            
            # Print by dorm
            for dorm, members in sorted(dorm_groups.items(), key=lambda x: len(x[1]), reverse=True):
                print(f"  [{dorm}] {len(members)} people: {', '.join(members)}")
        
        # Check for unassigned
        assigned = set()
        for passengers in allocation.values():
            assigned.update(passengers)
        unassigned = self.all_people - assigned
        
        if unassigned:
            print(f"\n⚠️  WARNING: {len(unassigned)} unassigned people:")
            for person in sorted(unassigned):
                print(f"    - {person} ({self.people_groups[person]})")
    
    def export_results(self, allocation: Dict, prefix="vehicle_allocation"):
        """Export allocation to CSV files."""
        # Detailed passenger list
        rows = []
        for (vehicle, driver), passengers in allocation.items():
            for passenger in passengers:
                rows.append({
                    'Vehicle': vehicle,
                    'Driver': driver,
                    'Passenger': passenger,
                    'Dorm': self.people_groups[passenger],
                    'Is_Driver': passenger == driver,
                    'Vehicle_Type': 'Minibus' if self.vehicles[vehicle]['is_minibus'] else 'Car'
                })
        
        detailed_df = pd.DataFrame(rows)
        detailed_df.to_csv(f'{prefix}_detailed.csv', index=False)
        
        # Summary by vehicle
        summary_rows = []
        for (vehicle, driver), passengers in allocation.items():
            dorm_counts = Counter(self.people_groups[p] for p in passengers)
            
            summary_rows.append({
                'Vehicle': vehicle,
                'Driver': driver,
                'Type': 'Minibus' if self.vehicles[vehicle]['is_minibus'] else 'Car',
                'Capacity': self.vehicles[vehicle]['capacity'] + 1,
                'Passengers': len(passengers),
                'Utilization_%': round(100 * len(passengers) / (self.vehicles[vehicle]['capacity'] + 1), 1),
                'Score': self.score_vehicle(passengers),
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
        print("CAMP VEHICLE ALLOCATION SYSTEM")
        print("Using 3-point seat belts only")
        print("Data folder: ./data/")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Show dorm summary
        print("\n" + "-"*40)
        print("DORM SUMMARY:")
        print("-"*40)
        dorm_summary = self.get_dorm_summary()
        print(dorm_summary.to_string())
        
        # Show vehicle summary
        print("\n" + "-"*40)
        print("VEHICLE SUMMARY:")
        print("-"*40)
        total_capacity = sum(v['capacity'] + 1 for v in self.vehicles.values())
        minibus_count = sum(1 for v in self.vehicles.values() if v['is_minibus'])
        car_count = len(self.vehicles) - minibus_count
        
        print(f"Total vehicles: {len(self.vehicles)}")
        print(f"  • Minibuses: {minibus_count}")
        print(f"  • Cars: {car_count}")
        print(f"Total capacity: {total_capacity} (using 3-point belts only)")
        print(f"Total people: {len(self.all_people)}")
        print(f"Spare capacity: {total_capacity - len(self.all_people)}")
        
        # Initial allocation
        print("\n" + "="*60)
        print("INITIAL ALLOCATION")
        print("="*60)
        
        allocation = self.allocate_vehicles()
        self.print_allocation(allocation)
        
        # Optimize
        print("\n" + "="*60)
        print("OPTIMIZING ALLOCATION")
        print("="*60)
        
        optimized = self.optimize_allocation(allocation, iterations=3000)
        self.print_allocation(optimized)
        
        # Export
        self.export_results(optimized)
        
        return optimized


def main():
    """Main entry point."""
    allocator = CampVehicleAllocator()
    allocator.run()


if __name__ == "__main__":
    main()