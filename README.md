# Camp Vehicle Allocation System

## Overview

This system optimally assigns campers to vehicles for transportation to camp. It uses mathematical optimization to find the best possible allocation that keeps dorm groups together while respecting vehicle capacities and driver assignments.

## 🎯 Goals

The system aims to:
1. **Keep dorm members together** - Minimize splitting dorms across multiple vehicles
2. **Avoid isolation** - Prevent anyone from being the only person from their dorm in a vehicle
3. **Ensure safety** - Respect vehicle capacities and assign appropriate supervision
4. **Use resources efficiently** - Only use as many vehicles as needed

## 📊 Input Data Required

### 1. Campers File (`campers.csv`)
Must contain:
- **Name**: Full name of each camper
- **Dorm**: Which dorm they belong to
- **Is Leader**: TRUE/FALSE - whether they're a leader
- **Gender**: M/F (optional - used for gender cohesion scoring)

### 2. Vehicles File (`cars.csv`)
Must contain:
- **Surname & Car Reg**: Vehicle identifier (e.g., "Minibus (XXXX YYY)")
- **3-point belts (excluding driver)**: Number of safe passenger seats
- **Driver 1, Driver 2, etc.**: Names of people authorized to drive this vehicle

## 🚦 Constraints & Rules

### Hard Requirements (Must Be Satisfied)
1. **Everyone gets a seat** - All campers must be assigned to exactly one vehicle
2. **Vehicle capacity** - Cannot exceed the number of 3-point seat belts
3. **Valid drivers only** - Vehicles can only be driven by authorized drivers listed in the data
4. **One driver per vehicle** - Each vehicle needs exactly one driver
5. **Drivers drive one vehicle** - A person can only drive one vehicle
6. **Minibus supervision** - Minibuses must have at least 2 leaders on board

### Soft Preferences (Optimizer Tries To Achieve)
1. **Minimum group size ≥ 3** - Avoid having less than 3 people from any dorm in a vehicle
2. **Single-dorm cars** - When possible, fill smaller cars with people from just one dorm
3. **Gender cohesion** - For mixed-dorm vehicles, prefer all boys or all girls

## 📈 Scoring System

The optimizer uses a points system to find the best allocation:

### Base Scores (Minimum Group Size)
The foundation of scoring is the **minimum group size** - the smallest number of people from any single dorm in a vehicle:

- **Size 1**: 1 point ⚠️ (heavily penalized)
- **Size 2**: 4 points ⚠️ (penalized)
- **Size 3**: 9 points ✓
- **Size 4**: 16 points ✓
- **Size 5**: 25 points ✓
- **Size 6**: 36 points ✓
- (Pattern continues: size² points)

### Bonuses
- **Single-dorm car**: +20 points (only for regular cars, not minibuses)
- **Gender cohesion**: +10 points (mixed-dorm vehicle with all same gender)

### Penalties
- **Minimum size 2**: -500 points
- **Minimum size 1**: -2000 points (someone isolated from their dorm)

### Example Scoring

**Good allocation** - Car with 5 people all from "Joe Bloggs" dorm:
- Base score: 25 (size 5)
- Single-dorm bonus: +20
- **Total: 45 points**

**Poor allocation** - Minibus with 3 from "Joe Bloggs", 2 from "Jane Doe", 1 from "Bob Jones":
- Base score: 1 (size 1 - someone isolated)
- Penalty: -2000
- **Total: -1999 points**

## 🔄 How the System Works

1. **Loads your data** - Reads camper and vehicle information
2. **Checks feasibility** - Ensures there are enough seats and drivers
3. **Sets up the problem** - Creates mathematical constraints
4. **Optimizes** - Searches millions of possible combinations to find the best one
5. **Outputs results** - Provides detailed allocation with scoring breakdown

## ⚠️ Important Notes

### Vehicles Excluded
- Vehicles with **0 three-point belts** are automatically excluded
- Vehicles with **no valid drivers** from the camper list cannot be used

### When No Perfect Solution Exists
If you have:
- Dorms with only 1-2 people
- Odd-numbered dorms (like 7 people)
- Not enough vehicles or drivers

The system will find the **best possible** solution, which might include some isolated individuals. It will clearly indicate when this happens.

### Time Limits
The optimizer runs for up to 5 minutes by default. For very complex scenarios with many campers and vehicles, it might need more time to find the absolute best solution.

## 📋 Reading the Output

The system provides:

### Summary Metrics
- **Total Score**: Higher is better (can be negative if many people are isolated)
- **Vehicles Used**: How many vehicles were needed
- **Warnings**: Any minibuses without enough leaders, people with min group size 1 or 2

### Vehicle Details
For each vehicle:
- **Driver**: Who's driving
- **Seats Used**: e.g., "12/17" means 12 of 17 seats used
- **Leaders**: Number of leaders on board
- **Min Group Size**: The smallest dorm group (key scoring factor)
- **Score Breakdown**: Points earned and penalties applied
- **Passenger List**: Organized by dorm with leaders marked (L)

### Example Output
```
Minibus (XXXX YYY)
  Driver: Joe Bloggs
  Seats: 15/17 used
  Leaders: 3
  Min group size: 4
  Vehicle score: 16 points
  [Joe Bloggs] 6 people: Joe Bloggs (L), Joe Bloggs 1, Joe Bloggs 2, ...
  [Jane Doe] 5 people: Jane Doe (L), Jane Doe 1, ...
  [Bob Jones] 4 people: Bob Jones (L), Bob Jones 1, ...
```

## 💡 Tips for Better Allocations

1. **Check your data** - Ensure all drivers in vehicles file appear in campers file
2. **Balance dorm sizes** - Very small dorms (1-2 people) make good allocations harder
3. **Sufficient leaders** - Need at least 2 leaders per minibus you plan to use
4. **Reasonable expectations** - With odd-numbered dorms, perfect allocations may be impossible

## 🆚 Manual vs. Optimized Allocations

You can compare your manual allocation with the optimized one using the same scoring system. This helps you understand:
- Where manual adjustments might help
- Why certain arrangements score better
- Trade-offs between different objectives

The optimizer considers millions of possibilities in seconds - something impossible to do manually - but human insight about specific camper needs can sometimes improve the final allocation.