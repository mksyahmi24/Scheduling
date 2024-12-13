import streamlit as st
import random
import pandas as pd

# Set the Streamlit page configuration
st.set_page_config(
    page_title="TV Scheduling Genetic Algorithm"
)

st.header("TV Scheduling Genetic Algorithm", divider="gray")

# Function to read the CSV file and convert it to the desired format
def read_csv_to_dict(file):
    program_ratings = {}
    df = pd.read_csv(file)
    for index, row in df.iterrows():
        program = row[0]
        ratings = row[1:].values.tolist()  # Convert the ratings to a list
        program_ratings[program] = ratings
    return program_ratings

# Crossover operation
def crossover(schedule1, schedule2):
    crossover_point = random.randint(1, len(schedule1) - 2)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2

# Mutation operation
def mutate(schedule, all_programs):
    mutation_point = random.randint(0, len(schedule) - 1)
    new_program = random.choice(all_programs)
    schedule[mutation_point] = new_program
    return schedule

# Ensure schedule matches the number of time slots
def validate_schedule_length(schedule, time_slots, all_programs):
    if len(schedule) < len(time_slots):
        # Add random programs if the schedule is too short
        schedule.extend(random.choices(all_programs, k=len(time_slots) - len(schedule)))
    elif len(schedule) > len(time_slots):
        # Trim the schedule if it's too long
        schedule = schedule[:len(time_slots)]
    return schedule

# Streamlit input for file upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    # Process the uploaded CSV file
    program_ratings_dict = read_csv_to_dict(uploaded_file)

    # Extract program names and time slots
    all_programs = list(program_ratings_dict.keys())  # All programs
    all_time_slots = list(range(6, 24))  # Time slots (6:00 AM to 11:00 PM)

    # Default genetic algorithm parameters
    GEN = 100  # Number of generations
    POP = 50  # Population size
    CO_R = 0.8  # Default crossover rate
    MUT_R = 0.02  # Default mutation rate
    EL_S = 2  # Elitism size

    # Define the fitness function
    def fitness_function(schedule):
        total_rating = 0
        for time_slot, program in enumerate(schedule):
            total_rating += program_ratings_dict[program][time_slot]
        return total_rating

    # Genetic algorithm
    def genetic_algorithm(initial_schedule, generations=GEN, population_size=POP, crossover_rate=CO_R, mutation_rate=MUT_R, elitism_size=EL_S):
        population = [validate_schedule_length(initial_schedule.copy(), all_time_slots, all_programs)]
        for _ in range(population_size - 1):
            random_schedule = initial_schedule.copy()
            random.shuffle(random_schedule)
            population.append(validate_schedule_length(random_schedule, all_time_slots, all_programs))

        for generation in range(generations):
            new_population = []
            population.sort(key=lambda schedule: fitness_function(schedule), reverse=True)
            new_population.extend(population[:elitism_size])

            while len(new_population) < population_size:
                parent1, parent2 = random.choices(population, k=2)
                if random.random() < crossover_rate:
                    child1, child2 = crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                if random.random() < mutation_rate:
                    child1 = mutate(child1, all_programs)
                if random.random() < mutation_rate:
                    child2 = mutate(child2, all_programs)

                new_population.extend([
                    validate_schedule_length(child1, all_time_slots, all_programs),
                    validate_schedule_length(child2, all_time_slots, all_programs)
                ])

            population = new_population

        return population[0]

    # Streamlit input interface for parameters
    with st.form("my_form"):
        CO_R = st.number_input("Crossover Rate", min_value=0.0, max_value=0.95, value=0.8, step=0.01)  # Manual input for crossover rate
        MUT_R = st.number_input("Mutation Rate", min_value=0.01, max_value=0.05, value=0.02, step=0.01)  # Manual input for mutation rate
        calculate = st.form_submit_button("Calculate")

        if calculate:
            initial_schedule = all_programs.copy()
            random.shuffle(initial_schedule)  # Randomize initial schedule
            best_schedule = genetic_algorithm(initial_schedule, crossover_rate=CO_R, mutation_rate=MUT_R)

            # Ensure the schedule length matches the time slots
            best_schedule = validate_schedule_length(best_schedule, all_time_slots, all_programs)

            # Create a DataFrame for the schedule
            schedule_df = pd.DataFrame({
                "Time Slot": [f"{hour:02d}:00" for hour in all_time_slots],
                "Program": best_schedule
            })

            # Display the final schedule as a table
            st.write("Optimal TV Schedule:")
            st.table(schedule_df)

            # Display the total fitness (ratings)
            total_ratings = fitness_function(best_schedule)
            st.write(f"Total Ratings: {total_ratings}")
