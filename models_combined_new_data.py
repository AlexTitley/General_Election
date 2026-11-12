import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import random
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV





party_naming_convention = {
    "Conservative": ["Conservative", "Con", "Cons", "Conservatives", "Conservative and Unionist", "Tory", "Tories", "con_share"],
    "Labour": ["Labour", "Lab", "Labour and Co-operative", "Labour Co-op", "Lab Co-op", "Lab/Co-op", "lab_share"],
    "Liberal Democrat": ["Liberal Democrat", "Lib Dem", "Lib Dems", "LibDem", "LibDems", "LD", "LDem", "Liberal Democrats","ld_share"],
    "Reform UK": ["Reform UK", "Reform", "Ref", "reform_share", "UK Independence Party", "UKIP", "United Kingdom Independence Party","The Brexit Party"],
    "Green Party": ["Green Party", "Green", "Greens", "Grn", "Green Party of England and Wales", "GPEW", "Scottish Green Party", "Scottish Greens", "Scottish Green", "SGP", "Green Party Northern Ireland", "Green Party NI", "green_share"],
    "Scottish National Party": ["Scottish National Party", "SNP", "Scottish Nationalist Party", "snp_pc_share"],
    "Plaid Cymru": ["Plaid Cymru", "Plaid", "PC", "Plaid Cymru - The Party of Wales", "Plaid Cymru - Party of Wales", "snp_pc_share"],
    "Other": ["nan" ,"other_share", "Other", "Others", "Oth", "Other parties", "Minor parties", "Independent", "Independents", "Ind", "IND", "Democratic Unionist Party", "DUP","Sinn Féin", "Sinn Fein", "SF", "Democratic Unionists","Alba Party", "Alba", "Ulster Unionist Party", "UUP", "Ulster Unionists", "Social Democratic & Labour Party", "SDLP", "Alliance Party of Northern Ireland", "Alliance", "APNI", "Alliance NI", "Your Party", 'YP', "Aontú", "Aontu", "Traditional Unionist Voice", "TUV", "Speaker", "Spk", "Speaker seeking re-election", "TUSC", "OMRLP", "Monster Raving Loony", "CPA", "WPB", "SLP", "SSP", "WEP", "AWP", "PBP", "People Before Profit", "CPB", "SDP", "BNP", "British National Party", "Yorkshire", "Yorkshire Party", "English Democrats", "EDP", "Abolish the Welsh Assembly Party", "Alliance for Democracy and Freedom", "Alliance for Green Socialism", "Animal Welfare Party", "Ashfield Independents", "Blue Revolution", "British Democratic Party", "British Unionist Party B.U.P.", "Chesterfield And North Derbyshire Independents (CANDI)", "Christian Party, Proclaiming Christ's Lordship", "Christian Peoples Alliance", "Climate Party", "Common Good", "Communist Future", "Communist League Election Campaign", "Communist Party of Britain", "Confelicity", "Consensus", "Count Binface Party", "Cross-Community Labour Alternative", "Democracy for Chorley", "English Constitution Party", "Fairer Voting Party", "Freedom Alliance", "Hampshire Independents", "Heritage Party", "Independence for Scotland Party", "Independent Alliance (Kent)", "Independent Network", "Independent Oxford Alliance", "Independents for Direct Democracy", "Kingston Independent Residents Group", "Liberal", "Libertarian Party", "Lincolnshire Independents Lincolnshire First", "Liverpool Community Independents", "National Health Action Party", "New Open Non-Political Organised Leadership", "Newham Independents Party", "Official Monster Raving Loony Party", "One Leicester", "Party of Women", "People Before Profit Alliance", "Portsmouth Independent Party", "Propel", "Psychedelic Movement", "Putting Crewe First, Independent Residents Group", "Rebooting Democracy", "Rejoin EU", "Save Us Now", "Scottish Family Party", "Scottish Libertarian Party", "Scottish Socialist Party", "Shared Ground", "Social Democratic Party", "Social Justice party", "Socialist Equality Party", "Socialist Labour Party", "Sovereignty", "South Devon Alliance", "Stockport Fights Austerity No To Cuts", "Swale Independents", "Taking The Initiative Party", "The Common People", "The Mitre TW9", "The North East Party", "The Peace Party - Non-violence, Justice, Environment", "The Socialist Party of Great Britain", "The Yorkshire Party", "The Yoruba Party in the UK", "Trade Unionist and Socialist Coalition", "Transform Party", "True & Fair Party", "UK Voice", "Volt United Kingdom", "Women's Equality Party", "Workers Party of Britain", "Workers Revolutionary Party","Young People's Party","Veterans and People's Party","The Best for Luton Party", "Advance Together","Heavy Woollen District Independents", "Birkenhead Social Justice Party", "Yeshua","Space Navies Party", "The Universal Good Party", "Renew","Church of the Militant Elvis Party", "The Independent Group for Change","Burnley and Padiham Independent Party", "Gwlad Gwlad", "Patria", "Advance Together","The Cynon Valley Party", "Communities United Party","The Justice & Anti-Corruption Party", "Young People's Party","The Citizens Movement Party UK", "Veterans and People's Party","Touch Love Worldwide (UK)", "The Best for Luton Party", "Shropshire Party","Proud of Oldham & Saddleworth", "Cumbria First","Mebyon Kernow - The Party for Cornwall", "Motherworld Party","Wycombe Independents", "The Constitution and Reform Party"],
}



###-----------------------------------------Defining Functions-----------------------------------------###
def get_index(ordered_data):
    start_index = 0
    input_list = pd.unique(ordered_data)
    index = []
    for x in input_list:
        for i in range(start_index, len(ordered_data)):
            y = ordered_data[i]
            if y != x:
                index.append([start_index,i - start_index])
                start_index = i
                break
            else:
                pass
    index.append([start_index,len(ordered_data)])
    return index

def get_seats(dataset):
    constituencies_raw, voteshare_raw, party_raw = dataset[0], dataset[1], dataset[2]
    constituencies_index = get_index(constituencies_raw)
    seats = [party_raw[np.argmax(voteshare_raw[x[0]:x[0]+x[1]])+x[0]] for x in constituencies_index]
    unique, counts = np.unique(seats, return_counts=True)
    return unique, counts

def get_party_votes(dataset):
    parties_raw, votes_raw = dataset[2], dataset[1]
    party_votes = {}
    for party, votes in zip(parties_raw,votes_raw):
        try:
            party_votes.update({party : votes + party_votes[party]})
        except:
            party_votes.update({party: votes})
    return party_votes



def get_voteshare(votes_dict):
    total_votes = sum(votes_dict.values())
    return{party: votes / total_votes for party, votes in votes_dict.items()}

def normalise_party_names(party_list):
    party_list_norm = []
    for party_in in party_list:
        party_name = None
        for party in party_naming_convention:
            if party_in in party_naming_convention[party]:
                party_name = party
                break
        if party_name != None:
            party_list_norm.append(party_name)
        elif pd.isnull(party_in) == True:
            party_list_norm.append("Other")
        else:
            print("Manual Review: ", party_in)
            party_list_norm.append(party_in)
    return(party_list_norm)


def Uniform_Swing(prev_election_dataset, forcast_dataset,forecast_sample_size, raw=False, local_sd= 0.05, systematic_sd=0.02): #sd_vars=(0.01,0.2) sd_vars = (0,0)
    prev_election_votes = get_party_votes(prev_election_dataset)
    forecast_votes = get_party_votes(forcast_dataset)
    prev_election_voteshare = get_voteshare(prev_election_votes)
    forecast_voteshare = get_voteshare(forecast_votes)
    voteshare_difference = {key: forecast_voteshare.get(key,0) -  prev_election_voteshare.get(key,0) for key in prev_election_voteshare.keys() | forecast_voteshare.keys()}
    indexes = get_index(prev_election_dataset[0])
    local_voteshare = np.concatenate([prev_election_dataset[1,index[0]:index[0] + index[1]] / sum(prev_election_dataset[1,index[0]:index[0] + index[1]]) for index in indexes])
    shifted_voteshare = np.clip(np.array([local_voteshare[x] + voteshare_difference[prev_election_dataset[2,x]] for x in range(len(prev_election_dataset[1]))]),a_min=0,a_max=None)
    national_sampling_error = {party: np.sqrt((voteshare *(1-voteshare)/ forecast_sample_size) + systematic_sd ** 2) for party,voteshare in forecast_voteshare.items()}                                                        #https://eprints.soton.ac.uk/id/eprint/421260/1/JenningsWlezien_PollingErrors.pdf
    if raw == True:
        stacked_output = [[prev_election_dataset[0, index[0]], [shifted_voteshare[index[0]:index[0] + index[1]], np.full(np.shape(shifted_voteshare[index[0]:index[0]+index[1]]), local_sd)], list(prev_election_dataset[2, index[0]:index[0] + index[1]])] for index in indexes]
        return stacked_output,national_sampling_error
    else:
        output = np.stack((prev_election_dataset[0], shifted_voteshare, prev_election_dataset[2]))
        return get_seats(output)


def monte_carlo_raw(model_outputs,iterations,national_sd={}):
    names = np.concatenate([[constituency[0]] * len(constituency[2]) for constituency in model_outputs])
    parties = np.concatenate([constituency[2] for constituency in model_outputs])
    means = np.concatenate([constituency[1][0] for constituency in model_outputs])
    local_sd = np.concatenate([constituency[1][1] for constituency in model_outputs])
    party_list, party_index = np.unique(parties,return_inverse=True)
    national_voteshare = get_voteshare(get_party_votes(np.stack((names, means, parties), dtype=object)))
    parties_voteshare = np.array([national_voteshare.get(party, 0) for party in party_list])
    parties_national_sd = np.array([national_sd.get(party,0) for party in party_list])
    log_sd = np.minimum(parties_national_sd / np.maximum(parties_voteshare, 1e-9), 0.3)

    mc_raw = []
    for i in range(iterations):
        simulated_voteshare = parties_voteshare * np.exp(log_sd * np.random.randn(len(party_list)))
        national_shock_normalised = simulated_voteshare * parties_voteshare.sum() / simulated_voteshare.sum() - parties_voteshare
        voteshares = means + national_shock_normalised[party_index] + (local_sd * np.random.randn(len(means)))
        mc_raw.append(get_seats(np.stack((names, voteshares, parties), dtype=object)))
    return mc_raw

def year_based_time_series_cv(years, n_splits=3):
    """TimeSeriesSplit applied to election *years* rather than rows, so every fold
    trains on whole earlier elections and validates on the next whole election
    (no constituency rows from the same election leak across the split).
    Returns a list of (train_idx, test_idx) positional index pairs for GridSearchCV."""
    years = pd.Series(np.asarray(years))
    unique_years = np.sort(years.unique())
    splits = []
    for train_y, test_y in TimeSeriesSplit(n_splits=n_splits).split(unique_years):
        train_idx = np.where(years.isin(unique_years[train_y]))[0]
        test_idx = np.where(years.isin(unique_years[test_y]))[0]
        splits.append((train_idx, test_idx))
    return splits


PARAM_GRID = {
    'n_estimators': [200, 500],
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [1, 3, 5],
    'max_features': ['sqrt', 0.5, 1.0],
}


def random_forest(dataset,backtest=False):
    df = dataset
    # (y) - the target columns
    target_cols = ['con_share', 'lab_share', 'ld_share', 'snp_pc_share', 'reform_share', 'green_share', 'other_share']
    y = df[target_cols]
    party_names_normalised = normalise_party_names(target_cols)
    # Features to be used in the model - make sure to match it with both CSV files
    feature_cols = ['year', 'boundary_set', 'region_encoded', 'prev_winner_encoded',
                    'prev_con_share', 'prev_lab_share', 'prev_ld_share', 'prev_snp_pc_share',
                    'prev_reform_share', 'prev_green_share', 'prev_other_share', 'prev_margin',
                    'prev_rel_con', 'prev_rel_lab', 'prev_rel_ld', 'prev_rel_snp_pc',
                    'prev_rel_reform', 'prev_rel_green', 'prev_rel_other',
                    'nat_swing_con', 'nat_swing_lab', 'nat_swing_ld', 'nat_swing_snp_pc',
                    'nat_swing_reform', 'nat_swing_green', 'nat_swing_other',
                    'house_price_rel', 'income_rel', 'employment_rel',
                    'pop_0_17_rel', 'pop_18_24_rel', 'pop_25_34_rel',
                    'pop_35_49_rel', 'pop_50_64_rel', 'pop_65_plus_rel']
    X = df[feature_cols]
    # Time-aware split: the most recent election is held out as the final test set,
    # and hyperparameters are tuned only on the elections before it.
    holdout_year = df['year'].max()
    dev_mask = (df['year'] < holdout_year).to_numpy()
    X_dev, y_dev = X[dev_mask], y[dev_mask]
    X_test, y_test = X[~dev_mask], y[~dev_mask]

    # GridSearchCV with expanding-window (TimeSeriesSplit) folds over election years
    cv_splits = year_based_time_series_cv(df.loc[dev_mask, 'year'], n_splits=3)
    grid = GridSearchCV(
        # n_jobs=1 on the forest, parallelism is handled by GridSearchCV
        estimator=RandomForestRegressor(random_state=42, n_jobs=1),
        param_grid=PARAM_GRID,
        cv=cv_splits,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    grid.fit(X_dev, y_dev)
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV MAE (time-series folds, raw %-point shares): {-grid.best_score_:.4f}")
    model = grid.best_estimator_
    # Evaluate on the held-out election
    raw_test_preds = model.predict(X_test)
    test_preds_df = pd.DataFrame(raw_test_preds, columns=target_cols, index=X_test.index)
    # If reform/green party stood in the constituency then the share is multiplied by 1, if not stood then share is multiplied by 0
    if 'reform_stood' in X_test.columns:
        test_preds_df['reform_share'] = test_preds_df['reform_share'] * df.loc[X_test.index, 'reform_stood']
    if 'green_stood' in X_test.columns:
        test_preds_df['green_share'] = test_preds_df['green_share'] * df.loc[X_test.index, 'green_stood']
    # Normalising test predictions to sum to 1.0 (100%)
    test_preds_normalized = test_preds_df.div(test_preds_df.sum(axis=1), axis=0)
    # Calculating Mean Absolute Error (MAE) to use in monte carlo simulation
    mae = mean_absolute_error(y_test/100, test_preds_normalized)
    print(f"Mean Absolute Error on {holdout_year} hold-out: {mae:.4f} (share, 0-1 scale)")

    if backtest:
        backtest_shares = test_preds_normalized.to_numpy()
        backtest_names = (df.loc[X_test.index, 'name'] + df.loc[X_test.index, 'year'].astype(str)).to_numpy()
        backtest_output = [[backtest_names[x], [backtest_shares[x], np.full(len(target_cols), mae)], party_names_normalised] for x in range(len(backtest_shares))]




    # Refit the tuned model on every election (including the hold-out year) for the forecast
    model = RandomForestRegressor(**grid.best_params_, random_state=42, n_jobs=-1)
    model.fit(X, y)
    # Preparing data for the prediction year
    future_df = pd.read_csv("prediction_dataset_new.csv")
    future_df['region_encoded'] = encoded_region.transform(future_df['region'])
    future_df['prev_winner_encoded'] = encoded_prev_winner.transform(future_df['prev_winner'])
    X_future = future_df[feature_cols]
    X_future = X_future.fillna(0)
    # Prediction for next general election
    future_raw_preds = model.predict(X_future)
    future_preds_df = pd.DataFrame(future_raw_preds, columns=target_cols, index=future_df.index)
    # Normalising future predictions to sum to 1.0 (100%)
    party_names_out = [party_names_normalised for x in range(df.shape[0])]
    future_preds_normalized = future_preds_df.div(future_preds_df.sum(axis=1), axis=0).to_numpy()
    #prep for return

    future_preds_variance = np.zeros(np.shape(future_preds_normalized))
    constituency_names = future_df['name'] + future_df['year'].astype(str)
    output = [[constituency_names[x], [future_preds_normalized[x], np.full(np.shape(future_preds_normalized[x]), mae)],party_names_normalised] for x in range(len(future_preds_normalized))]

    if backtest:
        return(backtest_output)
    else:
        return(output)


###-----------------------------------------Importing Data-----------------------------------------###
GE24 = pd.read_csv('candidate-level-results-general-election-04-07-2024.csv')
GE24_clean = np.stack((GE24['Constituency name'], GE24['Candidate vote count'] ,normalise_party_names(GE24['Main party name'])))



GE19 = pd.read_csv('HoC-GE2019-results-by-candidate.csv')
GE19_clean = np.stack((GE19['Constituency name'], GE19['Votes'] ,normalise_party_names(GE19['Party name'])))




YouGov_all_adults = pd.read_excel('voting-intention (1).xlsx', sheet_name='All adults')
YouGov_party_names = normalise_party_names(YouGov_all_adults['Party'].iloc[:-2])
YouGov_all_adults_clean = np.stack((np.zeros(len(YouGov_party_names),),np.rint(YouGov_all_adults['2026-10-19'].iloc[:-2]*YouGov_all_adults['2026-10-19'].iloc[-2]),YouGov_party_names),dtype=object)
YouGov_samplesize = YouGov_all_adults['2026-10-19'].iloc[-1]


YouGov_19_24_all_adults = pd.read_excel('voting-intention-2019-2024 (1).xlsx', sheet_name='All adults')
YouGov_19_24_party_names = normalise_party_names(YouGov_19_24_all_adults['If there were a general election held tomorrow, which party would you vote for?'].iloc[:-2])
YouGov_19_24_all_adults_clean = np.stack((np.zeros(len(YouGov_19_24_party_names),),np.rint(YouGov_19_24_all_adults['2024-07-03'].iloc[:-2]*YouGov_19_24_all_adults['2024-07-03'].iloc[-2]),YouGov_19_24_party_names),dtype=object)
YouGov_19_24_samplesize = YouGov_19_24_all_adults['2024-07-03'].iloc[-1]




collected_data = pd.read_csv("train_test_dataset_new.csv")                                       # Preparing the data into X feature matrix and y target vectors
collected_data = collected_data.fillna(0)                                                                # Filling in missing values
encoded_region = LabelEncoder()                                                  # Turning the string data to numerical values for the model to use
collected_data['region_encoded'] = encoded_region.fit_transform(collected_data['region'])
encoded_prev_winner = LabelEncoder()
collected_data['prev_winner_encoded'] = encoded_prev_winner.fit_transform(collected_data['prev_winner'])
###-----------------------------------------Main-----------------------------------------###

TESTS = []

TEST_SET = random_forest(collected_data,backtest=True)[-631:]
TESTS.append(monte_carlo_raw(TEST_SET,1000))

TEST_SET = random_forest(collected_data)                      # RF forecast for 2026 (prediction_dataset_new.csv)
TESTS.append(monte_carlo_raw(TEST_SET,1000))

TEST_SET, NATIONAL_SDS = Uniform_Swing(GE24_clean, YouGov_all_adults_clean,YouGov_samplesize,  raw=True)
TESTS.append(monte_carlo_raw(TEST_SET,1000,national_sd=NATIONAL_SDS))
#print(Uniform_Swing(GE24_clean,YouGov_all_adults_clean,raw=True))

TEST_SET, NATIONAL_SDS = Uniform_Swing(GE19_clean, YouGov_19_24_all_adults_clean,YouGov_19_24_samplesize,  raw=True)
TESTS.append(monte_carlo_raw(TEST_SET,1000,national_sd=NATIONAL_SDS))
###-----------------------------------------Metrics (temporary AI-generated block)-----------------------------------------###

MODEL_NAMES = ["RF 2024 backtest", "RF 2026 forecast (new data)", "UNS: 2024 result + YouGov Oct 2026", "UNS: 2019 result + YouGov Jul 2024 (backtest)"]   # same order as TESTS
ACTUAL_2024 = {"Labour": 411, "Conservative": 121, "Liberal Democrat": 72, "Reform UK": 5,
               "Scottish National Party": 9, "Plaid Cymru": 4, "Green Party": 4}   # only used for backtest models
LO, HI = 5, 95                                                                     # interval percentiles

for TEST, name in zip(TESTS, MODEL_NAMES):
    parties = sorted({p for u, _ in TEST for p in u})
    dicts = [dict(zip(u, c)) for u, c in TEST]
    seats = np.array([[d.get(p, 0) for p in parties] for d in dicts])        # simulations x parties
    total = int(seats[0].sum()); majority = total // 2 + 1
    is_backtest = "backtest" in name
    print(f"\n{'=' * 78}\n{name}  ({len(seats)} sims, {total} seats, majority {majority})\n{'=' * 78}")

    # 1. seat distribution per party (percentiles instead of min/max); actual result + its percentile for backtests
    print(f"{'Party':27s}{'mean':>6}{'median':>8}{f'{LO}%':>6}{f'{HI}%':>6}{'sd':>6}" + (f"{'actual':>8}{'pctile':>8}" if is_backtest else ""))
    for i in np.argsort(-np.median(seats, axis=0)):
        p = parties[i]
        if seats[:, i].mean() < 0.5 and not (is_backtest and p in ACTUAL_2024):
            continue
        lo, med, hi = np.percentile(seats[:, i], [LO, 50, HI])
        row = f"{p:27s}{seats[:, i].mean():6.0f}{med:8.0f}{lo:6.0f}{hi:6.0f}{seats[:, i].std():6.1f}"
        if is_backtest and p in ACTUAL_2024:
            a = ACTUAL_2024[p]
            row += f"{a:8d}{((seats[:, i] < a).mean() + 0.5 * (seats[:, i] == a).mean()) * 100:7.0f}%"
        print(row)

    # 2. largest party, single-party majority, hung parliament
    largest = seats.argmax(axis=1)
    print("\nLargest party: " + ", ".join(f"{parties[i]} {(largest == i).mean():.1%}" for i in range(len(parties)) if (largest == i).any()))
    has_majority = seats >= majority
    print("Single-party majority: " + (", ".join(f"{parties[i]} {has_majority[:, i].mean():.1%}" for i in range(len(parties)) if has_majority[:, i].any()) or "none"))
    print(f"Hung parliament: {(~has_majority.any(axis=1)).mean():.1%}")




###-----------------------------------------Vibecoded Graphs-----------------------------------------###
ORDER = ["Labour", "Conservative", "Liberal Democrat", "Reform UK",
         "Scottish National Party", "Others & NI", "Plaid Cymru", "Green Party"]
COLOURS = {"Labour": "#C8374A", "Conservative": "#4A84D4", "Liberal Democrat": "#E8A844",
           "Reform UK": "#6BB5CC", "Scottish National Party": "#CBA843",
           "Others & NI": "#9EA3AB", "Plaid Cymru": "#468B78", "Green Party": "#5BAA6A"}
MAJOR = {"Labour", "Conservative", "Liberal Democrat", "Reform UK", "Scottish National Party",
         "Plaid Cymru", "Green Party"}

def group_seats(parties, seats):
    """Merge every party not in MAJOR into 'Others & NI'."""
    out = {}
    for p, n in zip(parties, seats):
        key = p if p in MAJOR else "Others & NI"
        out[key] = out.get(key, 0) + int(n)
    return out

def plot_parliament(result, baseline=None, title="Uniform National Swing (UNS) projection",
                    subtitle="2024 general election results + YouGov poll (19/10/2026)", majority=326):
    now = group_seats(*result)
    before = group_seats(*baseline) if baseline else {}
    names = [p for p in ORDER if now.get(p, 0) > 0]
    seats = [now[p] for p in names]
    total = sum(seats)

    fig, ax = plt.subplots(figsize=(14, 3.2), facecolor="white")
    left = 0
    for p, n in zip(names, seats):
        ax.barh(0, n, left=left, color=COLOURS[p], edgecolor="black", linewidth=0.8, height=0.6)
        if n >= 15:
            ax.text(left + n / 2, 0, n, ha="center", va="center", color="white", fontweight="bold")
        left += n

    ax.axvline(majority, ymin=0.1, ymax=0.88, color="black", linewidth=1.5)
    ax.text(majority, 0.43, f"{majority} FOR A MAJORITY", ha="center", fontsize=8, fontweight="bold")
    ax.set_xlim(0, total)
    ax.set_ylim(-0.45, 0.55)
    ax.axis("off")

    labels = []
    for p, n in zip(names, seats):
        change = f" ({n - before.get(p, 0):+d})" if baseline else ""
        labels.append(f"{p} {n}{change}")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=COLOURS[p]) for p in names], labels,
              loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=False, fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.5, 0.88, subtitle, ha="center", fontsize=9, color="#555555")
    fig.subplots_adjust(top=0.78, bottom=0.25, left=0.02, right=0.98)
    plt.show()

#plot_parliament(Uniform_Swing(GE24_clean, YouGov_all_adults_clean), baseline=get_seats(GE24_clean))

