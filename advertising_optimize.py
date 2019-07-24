from tqdm import tqdm
from utils.genetic_algorithm import *
import argparse

## Population size of 100:
# Generate initial population
# For subsequent generations, select best 10 as parents,
# Crossover parents to generate 70 childs,
# Mutation on newly generated 70 childs,
# Generate another 20 new

## Constraints
# Limit of $300 budget a day - Done
# Start and end time between 0:00 and 24:00 - Done
# Each banner must be shown in one continuous block
# banner of 0 means nothing shown
# Choose from one of six banners per website

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--number_of_generations', type=int, default=100, help='number of generations')
    opt = parser.parse_args()

    ## Initial Generation
    marketing_plans = generate_marketing_plans(100)

    marketing_plans['cost'] = calculate_costs(marketing_plans)
    marketing_plans['predicted_clicks'] = predict_clicks(marketing_plans)

    best_marketing_plan_df = pd.DataFrame()
    best_marketing_plan_df = best_marketing_plan_df.append(marketing_plans.sort_values('predicted_clicks', ascending=False).head(1))

    ## Subsequent Generations
    for i in tqdm(range((opt.number_of_generations-1))):
        if i == 0:
            parents = marketing_plans.sort_values('predicted_clicks', ascending=False).head(10).copy()
            new_childs = crossover(parents, 70)
            scrambled_new_childs = mutate_scramble(new_childs, 0.1)
            scrambled_gaussian_new_childs = mutate_gaussian(scrambled_new_childs, 0.1)
            subsequent_generated_marketing_plans = generate_marketing_plans(20)
            subsequent_generated_marketing_plans['cost'] = calculate_costs(subsequent_generated_marketing_plans)
            subsequent_generated_marketing_plans['predicted_clicks'] = predict_clicks(subsequent_generated_marketing_plans)

            next_gen_marketing_plans = pd.concat([parents, scrambled_gaussian_new_childs, subsequent_generated_marketing_plans], axis=0)
        else:
            parents = next_gen_marketing_plans.sort_values('predicted_clicks', ascending=False).head(10).copy()
            new_childs = crossover(parents, 70)
            scrambled_new_childs = mutate_scramble(new_childs, 0.1)
            scrambled_gaussian_new_childs = mutate_gaussian(scrambled_new_childs, 0.1)
            subsequent_generated_marketing_plans = generate_marketing_plans(20)
            subsequent_generated_marketing_plans['cost'] = calculate_costs(subsequent_generated_marketing_plans)
            subsequent_generated_marketing_plans['predicted_clicks'] = predict_clicks(subsequent_generated_marketing_plans)

            next_gen_marketing_plans = pd.concat([parents, scrambled_gaussian_new_childs, subsequent_generated_marketing_plans], axis=0)

        best_marketing_plan_df = best_marketing_plan_df.append(next_gen_marketing_plans.sort_values('predicted_clicks', ascending=False).head(1))


    plot = best_marketing_plan_df['predicted_clicks'].reset_index(drop=True).plot(kind='line')
    fig = plot.get_figure()
    fig.savefig("output/fitness_across_{}_generations.png".format(str(opt.number_of_generations)))

    best_marketing_plan_df.to_csv('output/best_marketing_plan_across_{}_generations.csv'.format(str(opt.number_of_generations)), index=False)

#  best_marketing_plan_df.sort_values('predicted_clicks', ascending=False).head(1)
#ad0  ad1  ad2  ad3  ad4  ad0_start_time  ad1_start_time  ad2_start_time  ad3_start_time  ad4_start_time  ad0_time_spent  ad1_time_spent  ad2_time_spent  ad3_time_spent  ad4_time_spent   cost  predicted_clicks
#  3    0    6    1    5            23.5             0.0             0.2             5.2            23.8             0.4             0.0            19.7            16.1             0.6  299.6      423606.39354