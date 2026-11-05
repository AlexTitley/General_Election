import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error




party_naming_convention = {
    "Conservative": ["Conservative", "Con", "Cons", "Conservatives", "Conservative and Unionist", "Tory", "Tories", "con_share"],
    "Labour": ["Labour", "Lab", "Labour and Co-operative", "Labour Co-op", "Lab Co-op", "Lab/Co-op", "lab_share"],
    "Liberal Democrat": ["Liberal Democrat", "Lib Dem", "Lib Dems", "LibDem", "LibDems", "LD", "LDem", "Liberal Democrats","ld_share"],
    "Reform UK": ["Reform UK", "Reform", "Ref", "reform_share"],
    "Green Party": ["Green Party", "Green", "Greens", "Grn", "Green Party of England and Wales", "GPEW", "Scottish Green Party", "Scottish Greens", "Scottish Green", "SGP", "Green Party Northern Ireland", "Green Party NI", "green_share"],
    "Scottish National Party": ["Scottish National Party", "SNP", "Scottish Nationalist Party", "snp_pc_share"],
    "Plaid Cymru": ["Plaid Cymru", "Plaid", "PC", "Plaid Cymru - The Party of Wales", "Plaid Cymru - Party of Wales", "snp_pc_share"],
    "UK Independence Party": ["UK Independence Party", "UKIP", "United Kingdom Independence Party"],
    "Sinn Féin": ["Sinn Féin", "Sinn Fein", "SF"],
    "Democratic Unionist Party": ["Democratic Unionist Party", "DUP", "Democratic Unionists"],
    "Other": ["nan" ,"other_share", "Other", "Others", "Oth", "Other parties", "Minor parties", "Independent", "Independents", "Ind", "IND", "Alba Party", "Alba", "Ulster Unionist Party", "UUP", "Ulster Unionists", "Social Democratic & Labour Party", "SDLP", "Alliance Party of Northern Ireland", "Alliance", "APNI", "Alliance NI", "Your Party", 'YP', "Aontú", "Aontu", "Traditional Unionist Voice", "TUV", "Speaker", "Spk", "Speaker seeking re-election", "TUSC", "OMRLP", "Monster Raving Loony", "CPA", "WPB", "SLP", "SSP", "WEP", "AWP", "PBP", "People Before Profit", "CPB", "SDP", "BNP", "British National Party", "Yorkshire", "Yorkshire Party", "English Democrats", "EDP", "Abolish the Welsh Assembly Party", "Alliance for Democracy and Freedom", "Alliance for Green Socialism", "Animal Welfare Party", "Ashfield Independents", "Blue Revolution", "British Democratic Party", "British Unionist Party B.U.P.", "Chesterfield And North Derbyshire Independents (CANDI)", "Christian Party, Proclaiming Christ's Lordship", "Christian Peoples Alliance", "Climate Party", "Common Good", "Communist Future", "Communist League Election Campaign", "Communist Party of Britain", "Confelicity", "Consensus", "Count Binface Party", "Cross-Community Labour Alternative", "Democracy for Chorley", "English Constitution Party", "Fairer Voting Party", "Freedom Alliance", "Hampshire Independents", "Heritage Party", "Independence for Scotland Party", "Independent Alliance (Kent)", "Independent Network", "Independent Oxford Alliance", "Independents for Direct Democracy", "Kingston Independent Residents Group", "Liberal", "Libertarian Party", "Lincolnshire Independents Lincolnshire First", "Liverpool Community Independents", "National Health Action Party", "New Open Non-Political Organised Leadership", "Newham Independents Party", "Official Monster Raving Loony Party", "One Leicester", "Party of Women", "People Before Profit Alliance", "Portsmouth Independent Party", "Propel", "Psychedelic Movement", "Putting Crewe First, Independent Residents Group", "Rebooting Democracy", "Rejoin EU", "Save Us Now", "Scottish Family Party", "Scottish Libertarian Party", "Scottish Socialist Party", "Shared Ground", "Social Democratic Party", "Social Justice party", "Socialist Equality Party", "Socialist Labour Party", "Sovereignty", "South Devon Alliance", "Stockport Fights Austerity No To Cuts", "Swale Independents", "Taking The Initiative Party", "The Common People", "The Mitre TW9", "The North East Party", "The Peace Party - Non-violence, Justice, Environment", "The Socialist Party of Great Britain", "The Yorkshire Party", "The Yoruba Party in the UK", "Trade Unionist and Socialist Coalition", "Transform Party", "True & Fair Party", "UK Voice", "Volt United Kingdom", "Women's Equality Party", "Workers Party of Britain", "Workers Revolutionary Party"],
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


def UNS(prev_election_dataset, forcast_dataset, raw=False, sd_vars = (0.01,0.2)):
    prev_election_votes = get_party_votes(prev_election_dataset)
    forecast_votes = get_party_votes(forcast_dataset)
    prev_election_voteshare = get_voteshare(prev_election_votes)
    forecast_voteshare = get_voteshare(forecast_votes)
    voteshare_difference = {key: forecast_voteshare.get(key,0) -  prev_election_voteshare.get(key,0) for key in prev_election_voteshare.keys() | forecast_voteshare.keys()}
    indexes = get_index(prev_election_dataset[0])
    local_voteshare = np.concatenate([prev_election_dataset[1,index[0]:index[0] + index[1]] / sum(prev_election_dataset[1,index[0]:index[0] + index[1]]) for index in indexes])
    shifted_voteshare = np.clip(np.array([local_voteshare[x] + voteshare_difference[prev_election_dataset[2,x]] for x in range(len(prev_election_dataset[1]))]),a_min=0,a_max=None)
    if raw == True:
        stacked_GE29_UNS = [[prev_election_dataset[0, index[0]], [shifted_voteshare[index[0]:index[0] + index[1]], np.full((np.shape(shifted_voteshare[index[0]:index[0]+index[1]])), sd_vars[0]) + np.absolute(shifted_voteshare[index[0]:index[0] + index[1]] - local_voteshare[index[0]:index[0] + index[1]]) * sd_vars[1]], list(prev_election_dataset[2, index[0]:index[0] + index[1]])] for index in indexes]
        print(np.dtype(stacked_GE29_UNS))
        return stacked_GE29_UNS
    else:
        GE29_UNS = np.stack((prev_election_dataset[0], shifted_voteshare, prev_election_dataset[2]))
        return get_seats(GE29_UNS)


def monte_carlo_raw(model_outputs,iterations):
    names = np.concatenate([[constituency[0]] * len(constituency[2]) for constituency in model_outputs])
    parties = np.concatenate([constituency[2] for constituency in model_outputs])

    mc_raw = []
    for i in range(iterations):
        votes = np.concatenate([constituency[1][1] * np.random.randn(len(constituency[1][0])) + constituency[1][0]
                                for constituency in model_outputs])
        mc_raw.append(get_seats(np.stack((names, votes, parties), dtype=object)))
    return mc_raw

def random_forest(dataset):
    df = dataset
    # (y) - the target columns
    target_cols = ['con_share', 'lab_share', 'ld_share', 'snp_pc_share', 'reform_share', 'green_share', 'other_share']
    y = df[target_cols]
    # Features to be used in the model - make sure to match it with both CSV files
    feature_cols = ['year', 'boundary_set', 'region_encoded', 'prev_winner_encoded',
                    'prev_con_share', 'prev_lab_share', 'prev_ld_share', 'prev_snp_pc_share',
                    'prev_reform_share', 'prev_green_share', 'prev_other_share', 'prev_margin',
                    'prev_rel_con', 'prev_rel_lab', 'prev_rel_ld', 'prev_rel_snp_pc',
                    'prev_rel_reform', 'prev_rel_green', 'prev_rel_other',
                    'nat_swing_con', 'nat_swing_lab', 'nat_swing_ld', 'nat_swing_snp_pc',
                    'nat_swing_reform', 'nat_swing_green', 'nat_swing_other',
                    'house_price_rel', 'income_rel', 'employment_rel',
                    'pop_0_15_rel', 'pop_16_64_rel', 'pop_65_plus_rel']
    X = df[feature_cols]
    # Training and testing the model
    # Data split (80% for training, 20% for testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    # Evaluate on test set
    raw_test_preds = model.predict(X_test)
    test_preds_df = pd.DataFrame(raw_test_preds, columns=target_cols, index=X_test.index)
    # If reform/green party stood in the constituency then the share is multiplied by 1, if not stood then share is multiplied by 0
    if 'reform_stood' in X_test.columns:
        test_preds_df['reform_share'] = test_preds_df['reform_share'] * X_test['reform_stood']
    if 'green_stood' in X_test.columns:
        test_preds_df['green_share'] = test_preds_df['green_share'] * X_test['green_stood']
    # Normalising test predictions to sum to 1.0 (100%)
    test_preds_normalized = test_preds_df.div(test_preds_df.sum(axis=1), axis=0)
    # Calculating Mean Absolute Error (MAE) to use in monte carlo simulation
    mae = mean_absolute_error(y_test/100, test_preds_normalized)
    print(f"Mean Absolute Error: {mae:.4f} (percentage points)")
    # Preparing data for the prediction year
    future_df = pd.read_csv("prediction_dataset.csv")
    future_df['region_encoded'] = encoded_region.transform(future_df['region'])
    future_df['prev_winner_encoded'] = encoded_prev_winner.transform(future_df['prev_winner'])
    X_future = future_df[feature_cols]
    X_future = X_future.fillna(0)
    # Prediction for next general election
    future_raw_preds = model.predict(X_future)
    future_preds_df = pd.DataFrame(future_raw_preds, columns=target_cols, index=future_df.index)
    # Normalising future predictions to sum to 1.0 (100%)
    party_names_normalised = normalise_party_names(target_cols)
    party_names_out = [party_names_normalised for x in range(df.shape[0])]
    future_preds_normalized = future_preds_df.div(future_preds_df.sum(axis=1), axis=0).to_numpy()
    #prep for return

    future_preds_variance = np.zeros(np.shape(future_preds_normalized))
    constituency_names = df['name'] + df['year'].astype(str)
    '''
    print(len(constituency_names))
    print(len(future_preds_normalized))
    print(len(future_preds_variance))
    print(len(party_names_out))
    '''
    output = [[constituency_names[x], [future_preds_normalized[x], np.full(np.shape(future_preds_normalized[x]), mae)],party_names_normalised] for x in range(len(future_preds_normalized))]
    return(output)


###-----------------------------------------Importing Data-----------------------------------------###
GE24 = pd.read_csv('candidate-level-results-general-election-04-07-2024.csv')
GE24_clean = np.stack((GE24['Constituency name'], GE24['Candidate vote count'] ,normalise_party_names(GE24['Main party name'])))

YouGov_all_adults = pd.read_excel('voting-intention (1).xlsx', sheet_name='All adults')
YouGov_party_names = normalise_party_names(YouGov_all_adults['Party'].iloc[:-2])
YouGov_all_adults_clean = np.stack((np.zeros(len(YouGov_party_names),),np.rint(YouGov_all_adults['2026-10-19'].iloc[:-2]*YouGov_all_adults['2026-10-19'].iloc[-2]),YouGov_party_names),dtype=object)



collected_data = pd.read_csv("train_test_dataset.csv")                                       # Preparing the data into X feature matrix and y target vectors
collected_data = collected_data.fillna(0)                                                                # Filling in missing values
encoded_region = LabelEncoder()                                                  # Turning the string data to numerical values for the model to use
collected_data['region_encoded'] = encoded_region.fit_transform(collected_data['region'])
encoded_prev_winner = LabelEncoder()
collected_data['prev_winner_encoded'] = encoded_prev_winner.fit_transform(collected_data['prev_winner'])
###-----------------------------------------Main-----------------------------------------###



TEST_SET = random_forest(collected_data)[-631:]
TEST = monte_carlo_raw(TEST_SET,1000)




#print(UNS(GE24_clean,YouGov_all_adults_clean,raw=True))

#TEST_SET = UNS(GE24_clean, YouGov_all_adults_clean,raw=True)
#TEST = monte_carlo_raw(TEST_SET,1)
###-----------------------------------------Vibecoded Monte Carlo Metrics-----------------------------------------###
all_parties = sorted({party for constituency in TEST_SET for party in constituency[2]})
seat_counts = np.array([[dict(zip(u, c)).get(p, 0) for p in all_parties] for u, c in TEST])

# 1. Which parties win elections (largest party in each simulation)
winner_idx = seat_counts.argmax(axis=1)
wins = np.bincount(winner_idx, minlength=len(all_parties))
print("Elections won (largest party):")
for p, w in zip(all_parties, wins):
    if w > 0:
        print(f"  {p}: {w}")

print("  With a Majority: ", (seat_counts[:, all_parties.index("Labour")] >= 326).sum())

# 2. Min and max seats per party across all simulations
print("\nSeats across all simulations:")
for p, lo, hi in zip(all_parties, seat_counts.min(axis=0), seat_counts.max(axis=0)):
    print(f"  {p}: min {lo}, max {hi}")








'''
for i in range(0,10,1):
    for j in range(5,40,5):
        print("I:", i/100, "J:", j/100)
        UNS_TEST_SET = UNS(GE24_clean, YouGov_all_adults_clean,raw=True,sd_vars=(i/100,j/100))
        TEST = monte_carlo_raw(UNS_TEST_SET,1000)
        ###-----------------------------------------Vibecoded Monte Carlo Metrics-----------------------------------------###
        all_parties = sorted({party for constituency in UNS_TEST_SET for party in constituency[2]})
        seat_counts = np.array([[dict(zip(u, c)).get(p, 0) for p in all_parties] for u, c in TEST])

        # 1. Which parties win elections (largest party in each simulation)
        winner_idx = seat_counts.argmax(axis=1)
        wins = np.bincount(winner_idx, minlength=len(all_parties))
        print("Elections won (largest party):")
        for p, w in zip(all_parties, wins):
            if w > 0:
                print(f"  {p}: {w}")

        print("  With a Majority: ", (seat_counts[:, all_parties.index("Labour")] >= 326).sum())

        # 2. Min and max seats per party across all simulations
        print("\nSeats across all simulations:")
        for p, lo, hi in zip(all_parties, seat_counts.min(axis=0), seat_counts.max(axis=0)):
            print(f"  {p}: min {lo}, max {hi}")
'''














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

#plot_parliament(UNS(GE24_clean, YouGov_all_adults_clean), baseline=get_seats(GE24_clean))

