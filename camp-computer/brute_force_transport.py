import polars as pl
import time
from itertools import combinations, product
from collections import defaultdict

class TransportBruteForce:
    def __init__(self, people_file="data/people.csv", cars_file="data/cars.csv"):
        self.start_time = time.time()
        self.load_data(people_file, cars_file)
        self.best_score = -float('inf')
        self.best_allocation = None
        self.scenarios_tested = 0
        
    def load_data(self, people_file, cars_file):
        """Load and prepare all data"""
        print("Loading data...")
        
        # Load people
        people = pl.read_csv(people_file)
        not_going = ["Eleanor Clarke", "Anthony Bewes"]
        self.going_people = people.filter(~pl.col('Name').is_in(not_going))
        
        self.all_participants = self.going_people.select('Name').to_series().to_list()
        self.leaders = self.going_people.filter(pl.col('Is Leader') == True).select('Name').to_series().to_list()
        
        # Create lookup dictionaries
        self.participant_to_group = dict(zip(self.going_people.select('Name').to_series(), 
                                           self.going_people.select('Dorm').to_series()))
        self.participant_to_gender = dict(zip(self.going_people.select('Name').to_series(), 
                                            self.going_people.select('Sex').to_series()))
        
        # Load cars
        cars_df = pl.scan_csv(cars_file).with_columns(
            (pl.col('3-point belts (excluding driver)') + 1).alias('seats')
        ).collect()
        
        self.cars = cars_df.select('Surname & Car Reg').to_series().to_list()
        self.car_to_capacity = dict(zip(cars_df.select('Surname & Car Reg').to_series(), 
                                      (cars_df.select('seats').to_series() - 1)))
        
        # Build driver authorization mapping
        self.car_to_drivers = {}
        self.driver_to_cars = defaultdict(list)
        
        for row in cars_df.iter_rows(named=True):
            car = row['Surname & Car Reg']
            authorized_drivers = []
            
            for i in range(1, 7):
                driver = row.get(f'Driver {i}')
                if driver and driver in self.all_participants:
                    authorized_drivers.append(driver)
                    self.driver_to_cars[driver].append(car)
            
            self.car_to_drivers[car] = authorized_drivers
        
        # Identify special constraints
        self.minibus_cars = [car for car in self.cars if 'Minibus' in car]
        self.groups = self.going_people.select('Dorm').unique().to_series().to_list()
        
        print(f"Loaded: {len(self.all_participants)} people, {len(self.cars)} cars, {len(self.leaders)} leaders")
        print(f"Minibuses: {len(self.minibus_cars)}")
        
    def estimate_cars_needed(self):
        """Calculate minimum and reasonable maximum cars needed"""
        total_people = len(self.all_participants)
        
        # Key insight: drivers don't need passenger seats!
        # If we use 15 cars, we need 15 drivers, leaving 93-15=78 people needing passenger seats
        
        # Try different numbers of cars and see if passenger capacity is sufficient
        capacities = sorted(self.car_to_capacity.values(), reverse=True)
        
        min_cars = 0
        for num_cars in range(1, len(self.cars) + 1):
            # If we use num_cars, we need num_cars drivers
            passengers_needing_seats = total_people - num_cars
            
            # Get capacity of the largest num_cars cars
            total_passenger_capacity = sum(capacities[:num_cars])
            
            if total_passenger_capacity >= passengers_needing_seats:
                min_cars = num_cars
                break
        
        # Add some flexibility for optimization
        max_cars = min(len(self.cars), min_cars + 3)
        
        return min_cars, max_cars
    
    def get_valid_driver_assignments(self, car_subset):
        """Generate all valid driver assignments for a car subset"""
        
        # Separate cars by driver flexibility
        fixed_assignments = {}  # Car -> only possible driver
        flexible_cars = []      # Cars with multiple driver options
        
        for car in car_subset:
            possible_drivers = [d for d in self.car_to_drivers[car] if d in self.all_participants]
            
            if len(possible_drivers) == 0:
                return []  # No valid drivers - skip this subset
            elif len(possible_drivers) == 1:
                fixed_assignments[car] = possible_drivers[0]
            else:
                flexible_cars.append((car, possible_drivers))
        
        # Check if fixed assignments conflict (same person assigned to multiple cars)
        used_drivers = set()
        for car, driver in fixed_assignments.items():
            if driver in used_drivers:
                return []  # Conflict - same driver needed for multiple cars
            used_drivers.add(driver)
        
        # Generate all combinations for flexible cars
        if not flexible_cars:
            return [fixed_assignments] if fixed_assignments else [{}]
        
        valid_assignments = []
        
        # Get all possible driver combinations for flexible cars
        flexible_options = [drivers for car, drivers in flexible_cars]
        
        for driver_combo in product(*flexible_options):
            assignment = fixed_assignments.copy()
            valid = True
            
            # Check for conflicts with flexible assignments
            for i, (car, _) in enumerate(flexible_cars):
                driver = driver_combo[i]
                if driver in assignment.values():
                    valid = False  # Driver already assigned
                    break
                assignment[car] = driver
            
            if valid:
                valid_assignments.append(assignment)
        
        return valid_assignments
    
    def greedy_passenger_assignment(self, driver_assignments):
        """Assign passengers using greedy algorithm, then test all ways to fill remaining seats"""
        
        base_allocation = {
            'drivers': driver_assignments,
            'passengers': {car: [] for car in driver_assignments.keys()}
        }
        
        # Track available people (excluding drivers)
        available_people = [p for p in self.all_participants if p not in driver_assignments.values()]
        
        # Phase 1: Greedy assignment up to capacity-2
        # (Same as before but separated for clarity)
        base_allocation, remaining_people = self._greedy_to_capacity_minus_2(base_allocation, available_people)
        
        # Phase 2: Test all possible ways to fill/leave empty the remaining seats
        return self._optimize_remaining_seats(base_allocation, remaining_people)
    
    def _greedy_to_capacity_minus_2(self, allocation, available_people):
        """Greedy assignment up to capacity-2 for each car"""
        
        # Phase 1: Assign room group members to their leaders
        for car, driver in allocation['drivers'].items():
            driver_room = self.participant_to_group[driver]
            room_members = [p for p in available_people if self.participant_to_group[p] == driver_room]
            capacity = self.car_to_capacity[car]
            max_passengers = max(0, capacity - 2)  # Stop at capacity-2
            
            # Add room members up to max_passengers
            added = 0
            for member in room_members[:]:  # Copy to avoid modification issues
                if added < max_passengers:
                    allocation['passengers'][car].append(member)
                    available_people.remove(member)
                    added += 1
                else:
                    break
        
        # Phase 2: Continue adding people who can join someone from their room group
        changed = True
        while changed and available_people:
            changed = False
            
            for person in available_people[:]:
                person_room = self.participant_to_group[person]
                
                # Find cars with people from same room group and available space
                for car in allocation['passengers']:
                    capacity = self.car_to_capacity[car]
                    max_passengers = max(0, capacity - 2)  # Stop at capacity-2
                    
                    if len(allocation['passengers'][car]) < max_passengers:
                        # Check if car has someone from same room (including driver)
                        car_driver = allocation['drivers'][car]
                        car_passengers = allocation['passengers'][car]
                        
                        same_room_in_car = (
                            self.participant_to_group[car_driver] == person_room or
                            any(self.participant_to_group[p] == person_room for p in car_passengers)
                        )
                        
                        if same_room_in_car:
                            allocation['passengers'][car].append(person)
                            available_people.remove(person)
                            changed = True
                            break
                
                if changed:
                    break
        
        # Phase 3: Handle remaining people avoiding singletons (still up to capacity-2)
        for person in available_people[:]:
            person_room = self.participant_to_group[person]
            best_car = None
            best_score = -1
            
            for car in allocation['passengers']:
                capacity = self.car_to_capacity[car]
                max_passengers = max(0, capacity - 2)  # Stop at capacity-2
                
                if len(allocation['passengers'][car]) < max_passengers:
                    # Count people from person's room in this car
                    car_driver = allocation['drivers'][car]
                    car_passengers = allocation['passengers'][car]
                    
                    same_room_count = 0
                    if self.participant_to_group[car_driver] == person_room:
                        same_room_count += 1
                    same_room_count += sum(1 for p in car_passengers if self.participant_to_group[p] == person_room)
                    
                    if same_room_count > best_score:
                        best_score = same_room_count
                        best_car = car
            
            if best_car:
                allocation['passengers'][best_car].append(person)
                available_people.remove(person)
        
        return allocation, available_people
    
    def _optimize_remaining_seats(self, base_allocation, remaining_people):
        """Test all possible ways to fill the remaining 2 seats per car"""
        
        if not remaining_people:
            # No one left to assign
            base_allocation['unassigned'] = []
            return base_allocation
        
        # Get available seats (up to 2 per car)
        available_seats = []
        for car in base_allocation['drivers']:
            capacity = self.car_to_capacity[car]
            current_passengers = len(base_allocation['passengers'][car])
            remaining_seats = capacity - current_passengers
            
            # Add seat options for this car (0 to min(2, remaining_seats) additional people)
            for additional_people in range(min(remaining_seats + 1, 3)):  # 0, 1, or 2 additional
                available_seats.append((car, additional_people))
        
        # Generate all possible assignments of remaining people to available seats
        best_allocation = None
        best_score = -float('inf')
        
        # This is still manageable: ~17 cars × 3 options = ~51 choices per person
        # For 18 remaining people, this is computationally intensive but doable
        
        # For now, use a simpler heuristic: try to assign remaining people optimally
        final_allocation = self._heuristic_final_assignment(base_allocation, remaining_people)
        
        return final_allocation
    
    def _heuristic_final_assignment(self, allocation, remaining_people):
        """Smart heuristic for final seat assignment"""
        
        # Try to assign remaining people to cars where they won't be singletons
        for person in remaining_people[:]:
            person_room = self.participant_to_group[person]
            best_car = None
            best_score = -1
            
            for car in allocation['passengers']:
                capacity = self.car_to_capacity[car]
                current_passengers = len(allocation['passengers'][car])
                
                if current_passengers < capacity:  # Now we can fill to full capacity
                    # Count people from person's room in this car
                    car_driver = allocation['drivers'][car]
                    car_passengers = allocation['passengers'][car]
                    
                    same_room_count = 0
                    if self.participant_to_group[car_driver] == person_room:
                        same_room_count += 1
                    same_room_count += sum(1 for p in car_passengers if self.participant_to_group[p] == person_room)
                    
                    # Prefer cars where person won't be a singleton
                    score = same_room_count if same_room_count > 0 else -1
                    if score > best_score:
                        best_score = score
                        best_car = car
            
            if best_car and best_score >= 0:  # Only assign if not creating singleton
                allocation['passengers'][best_car].append(person)
                remaining_people.remove(person)
        
        allocation['unassigned'] = remaining_people
        return allocation
    
    def validate_allocation(self, allocation):
        """Check if allocation satisfies all hard constraints"""
        violations = []
        
        # NEW: Everyone must be assigned
        if len(allocation['unassigned']) > 0:
            violations.append(f"Unassigned people: {len(allocation['unassigned'])} people not assigned")
        
        # Check capacity constraints
        for car in allocation['passengers']:
            capacity = self.car_to_capacity[car]
            actual = len(allocation['passengers'][car])
            if actual > capacity:
                violations.append(f"Over capacity: {car} has {actual} passengers, capacity {capacity}")
        
        # Check minibus leader requirements
        for car in self.minibus_cars:
            if car in allocation['drivers']:
                driver = allocation['drivers'][car]
                passengers = allocation['passengers'][car]
                
                leader_count = 0
                if driver in self.leaders:
                    leader_count += 1
                leader_count += sum(1 for p in passengers if p in self.leaders)
                
                if leader_count < 2:
                    violations.append(f"Minibus {car} has only {leader_count} leaders, needs 2+")
        
        # Check gender mixing
        for car in allocation['drivers']:
            driver = allocation['drivers'][car]
            passengers = allocation['passengers'][car]
            
            if self.participant_to_gender.get(driver) == 'M':
                male_passengers = [p for p in passengers if self.participant_to_gender.get(p) == 'M']
                female_passengers = [p for p in passengers if self.participant_to_gender.get(p) == 'F']
                
                if female_passengers and not male_passengers:
                    violations.append(f"Male driver {driver} with only female passengers in {car}")
        
        # Check for singletons (min 2 people per room per car)
        for car in allocation['drivers']:
            driver = allocation['drivers'][car]
            passengers = allocation['passengers'][car]
            
            # Count people from each room in this car
            room_counts = defaultdict(int)
            room_counts[self.participant_to_group[driver]] += 1
            for passenger in passengers:
                room_counts[self.participant_to_group[passenger]] += 1
            
            # Check for singletons
            for room, count in room_counts.items():
                if count == 1:
                    violations.append(f"Singleton: Only 1 person from {room} in {car}")
        
        # Check no leaders driving alone (must have at least 1 passenger)
        for car in allocation['drivers']:
            driver = allocation['drivers'][car]
            passengers = allocation['passengers'][car]
            
            if driver in self.leaders and len(passengers) == 0:
                violations.append(f"Leader driving alone: {driver} in {car} with no passengers")
        
        return violations
    
    def score_allocation(self, allocation):
        """Score allocation using minimum group size scoring with leader penalties"""
        score = 0
        
        # Base score: 50 points per person assigned
        total_assigned = len(allocation['drivers']) + sum(len(passengers) for passengers in allocation['passengers'].values())
        score += total_assigned * 50
        
        # Minimum group size scoring for each car
        for car in allocation['passengers']:
            driver = allocation['drivers'][car]
            passengers = allocation['passengers'][car]
            
            if not passengers:  # Skip empty cars
                continue
            
            # Count people from each room group in this car (including driver)
            room_counts = defaultdict(int)
            room_counts[self.participant_to_group[driver]] += 1
            
            for passenger in passengers:
                room_counts[self.participant_to_group[passenger]] += 1
            
            # Find minimum group size (excluding groups with 0 people)
            group_sizes = [count for count in room_counts.values() if count > 0]
            min_group_size = min(group_sizes) if group_sizes else 0
            
            # Check if this is a pure room group car (only one room represented)
            is_pure_room_car = len([count for count in room_counts.values() if count > 0]) == 1
            
            # Check if driver is from the majority room group
            driver_room = self.participant_to_group[driver]  
            max_room_size = max(room_counts.values()) if room_counts else 0
            largest_rooms = [room for room, count in room_counts.items() if count == max_room_size]
            driver_leads_largest_group = driver_room in largest_rooms
            
            # Hierarchy-based scoring with leader considerations
            if min_group_size == 1:
                score += 100  # Everything else (worst)
            elif min_group_size == 2:
                score += 5000  # 2+2 or 4+2 scenarios
            elif min_group_size == 3:
                score += 10000  # 3+3 scenarios
            elif min_group_size >= 4:
                if is_pure_room_car and driver in self.leaders and driver_leads_largest_group:
                    # Perfect: pure room group with correct leader driving
                    score += 150000  # MASSIVE bonus for correct leader
                elif is_pure_room_car and driver_leads_largest_group:
                    # Good: pure room group with non-leader from same room driving
                    score += 100000  # Standard pure group bonus
                elif is_pure_room_car:
                    # Bad: pure room group but wrong driver from different room
                    score += 20000   # Heavy penalty - much worse than mixed cars
                else:
                    # Mixed car with min group size 4+
                    score += 100000  # Standard bonus
        
        # Additional leader driving own room bonus (for non-pure scenarios)
        for car in allocation['passengers']:
            driver = allocation['drivers'][car]  
            passengers = allocation['passengers'][car]
            
            if driver in self.leaders:
                driver_room = self.participant_to_group[driver]
                room_passengers = [p for p in passengers if self.participant_to_group[p] == driver_room]
                
                # Only give this bonus if not already covered by pure car scoring above
                room_counts = defaultdict(int)
                room_counts[self.participant_to_group[driver]] += 1
                for passenger in passengers:
                    room_counts[self.participant_to_group[passenger]] += 1
                
                is_pure_room_car = len([count for count in room_counts.values() if count > 0]) == 1
                
                if not is_pure_room_car:  # Only for mixed cars
                    score += len(room_passengers) * 3000  # 3000 per room member for mixed cars
        
        # Penalty for unassigned people
        score -= len(allocation['unassigned']) * 10000
        
        return score
    
    def optimize(self, max_scenarios=100000):
        """Run brute force optimization"""
        print("Starting brute force optimization...")
        
        min_cars, max_cars = self.estimate_cars_needed()
        print(f"Will test using {min_cars}-{max_cars} cars")
        
        total_scenarios = 0
        for num_cars in range(min_cars, max_cars + 1):
            car_subsets = list(combinations(self.cars, num_cars))
            total_scenarios += len(car_subsets)
        
        print(f"Estimated scenarios to test: {total_scenarios:,}")
        
        scenarios_tested = 0
        last_progress = time.time()
        
        for num_cars in range(min_cars, max_cars + 1):
            print(f"\nTesting with {num_cars} cars...")
            
            for car_subset in combinations(self.cars, num_cars):
                # Early termination if taking too long
                if scenarios_tested >= max_scenarios:
                    print(f"Reached maximum scenarios limit: {max_scenarios:,}")
                    break
                
                # Progress reporting
                scenarios_tested += 1
                if time.time() - last_progress > 30:  # Every 30 seconds
                    elapsed = time.time() - self.start_time
                    rate = scenarios_tested / elapsed
                    print(f"Progress: {scenarios_tested:,} scenarios tested, {rate:.0f}/sec, best score: {self.best_score:,}")
                    last_progress = time.time()
                
                # Get valid driver assignments for this car subset
                driver_assignments_list = self.get_valid_driver_assignments(car_subset)
                
                if not driver_assignments_list:
                    continue  # No valid driver assignments
                
                # Test each driver assignment
                for driver_assignments in driver_assignments_list:
                    # Check basic feasibility
                    total_capacity = sum(self.car_to_capacity[car] for car in car_subset)
                    if total_capacity < len(self.all_participants) - len(driver_assignments):
                        continue  # Not enough capacity
                    
                    # Run greedy passenger assignment
                    allocation = self.greedy_passenger_assignment(driver_assignments)
                    
                    # Validate constraints
                    violations = self.validate_allocation(allocation)
                    if violations:
                        continue  # Skip invalid allocations
                    
                    # Score allocation
                    score = self.score_allocation(allocation)
                    
                    if score > self.best_score:
                        self.best_score = score
                        self.best_allocation = allocation
                        print(f"NEW BEST: Score {score:,} with {num_cars} cars")
            
            if scenarios_tested >= max_scenarios:
                break
        
        elapsed = time.time() - self.start_time
        print(f"\nOptimization complete!")
        print(f"Scenarios tested: {scenarios_tested:,}")
        print(f"Time elapsed: {elapsed:.1f} seconds")
        print(f"Best score: {self.best_score:,}")
        
        return self.best_allocation
    
    def print_allocation(self, allocation):
        """Print allocation in readable format"""
        if not allocation:
            print("No allocation to display")
            return
        
        print("\n" + "="*60)
        print("BRUTE FORCE OPTIMAL ALLOCATION")
        print("="*60)
        
        cars_used = len(allocation['drivers'])
        total_passengers = sum(len(passengers) for passengers in allocation['passengers'].values())
        
        print(f"Cars used: {cars_used}")
        print(f"Total passengers: {total_passengers}")
        print(f"Unassigned: {len(allocation['unassigned'])}")
        
        for car in sorted(allocation['drivers'].keys()):
            driver = allocation['drivers'][car]
            passengers = allocation['passengers'][car]
            
            print(f"\n🚗 {car}")
            print(f"   Driver: {driver}")
            print(f"   Passengers ({len(passengers)}):")
            
            # Group by room
            passengers_by_room = defaultdict(list)
            for passenger in passengers:
                room = self.participant_to_group[passenger]
                passengers_by_room[room].append(passenger)
            
            for room, members in passengers_by_room.items():
                leader_count = sum(1 for m in members if m in self.leaders)
                leader_indicator = f" ({leader_count}L)" if leader_count > 0 else ""
                print(f"     {room}{leader_indicator}: {', '.join(sorted(members))}")
        
        if allocation['unassigned']:
            print(f"\n❌ UNASSIGNED ({len(allocation['unassigned'])}):")
            for person in allocation['unassigned']:
                print(f"   - {person}")

# Main execution
if __name__ == "__main__":
    optimizer = TransportBruteForce()
    
    # Run optimization with reasonable limits
    best_allocation = optimizer.optimize(max_scenarios=50000)  # Limit to prevent very long runs
    
    # Display results
    optimizer.print_allocation(best_allocation)