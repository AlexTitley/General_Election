###-----------------------------------------Plan-----------------------------------------###

'''
UNS Steps:
Data:
    Dataset format:
        [0] Constituency name
        [1] Candidate vote count
        [2] Main party name
    Dataset 1:
        General Election Results 2024
        Notes:
            7 seats won as NaN - 6 are independents 1 is the speaker
    Dataset 2:
        YouGov current polling(20/10/26)




Find %swing difference between them
    Formula:
        for each party:
            Swing = National Voteshare Previous - National Voteshare Predicted


Apply to all constituencies
    Formula:
        for each party:
            Local Voteshare Previous + Swing
    Notes:
        Has many possible errors like negative votes or voteshare above 100%

Calc seats
    Formula:
        Party with max(voteshare)
'''
from random import random

###-----------------------------------------Importing Libraries-----------------------------------------###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




###-----------------------------------------Defining Constants-----------------------------------------###
#Disclosure: Lists generated using ai
party_naming_convention = {
    "Conservative": ["Conservative", "Con", "Cons", "Conservatives", "Conservative and Unionist", "Tory", "Tories"],
    "Labour": ["Labour", "Lab", "Labour and Co-operative", "Labour Co-op", "Lab Co-op", "Lab/Co-op"],
    "Liberal Democrat": ["Liberal Democrat", "Lib Dem", "Lib Dems", "LibDem", "LibDems", "LD", "LDem", "Liberal Democrats"],
    "Reform UK": ["Reform UK", "Reform", "Ref"],
    "Green Party": ["Green Party", "Green", "Greens", "Grn", "Green Party of England and Wales", "GPEW", "Scottish Green Party", "Scottish Greens", "Scottish Green", "SGP", "Green Party Northern Ireland", "Green Party NI"],
    "Scottish National Party": ["Scottish National Party", "SNP", "Scottish Nationalist Party"],
    "Plaid Cymru": ["Plaid Cymru", "Plaid", "PC", "Plaid Cymru - The Party of Wales", "Plaid Cymru - Party of Wales"],
    "UK Independence Party": ["UK Independence Party", "UKIP", "United Kingdom Independence Party"],
    "Sinn Féin": ["Sinn Féin", "Sinn Fein", "SF"],
    "Democratic Unionist Party": ["Democratic Unionist Party", "DUP", "Democratic Unionists"],
    "Other": ["nan" ,"Other", "Others", "Oth", "Other parties", "Minor parties", "Independent", "Independents", "Ind", "IND", "Alba Party", "Alba", "Ulster Unionist Party", "UUP", "Ulster Unionists", "Social Democratic & Labour Party", "SDLP", "Alliance Party of Northern Ireland", "Alliance", "APNI", "Alliance NI", "Your Party", 'YP', "Aontú", "Aontu", "Traditional Unionist Voice", "TUV", "Speaker", "Spk", "Speaker seeking re-election", "TUSC", "OMRLP", "Monster Raving Loony", "CPA", "WPB", "SLP", "SSP", "WEP", "AWP", "PBP", "People Before Profit", "CPB", "SDP", "BNP", "British National Party", "Yorkshire", "Yorkshire Party", "English Democrats", "EDP", "Abolish the Welsh Assembly Party", "Alliance for Democracy and Freedom", "Alliance for Green Socialism", "Animal Welfare Party", "Ashfield Independents", "Blue Revolution", "British Democratic Party", "British Unionist Party B.U.P.", "Chesterfield And North Derbyshire Independents (CANDI)", "Christian Party, Proclaiming Christ's Lordship", "Christian Peoples Alliance", "Climate Party", "Common Good", "Communist Future", "Communist League Election Campaign", "Communist Party of Britain", "Confelicity", "Consensus", "Count Binface Party", "Cross-Community Labour Alternative", "Democracy for Chorley", "English Constitution Party", "Fairer Voting Party", "Freedom Alliance", "Hampshire Independents", "Heritage Party", "Independence for Scotland Party", "Independent Alliance (Kent)", "Independent Network", "Independent Oxford Alliance", "Independents for Direct Democracy", "Kingston Independent Residents Group", "Liberal", "Libertarian Party", "Lincolnshire Independents Lincolnshire First", "Liverpool Community Independents", "National Health Action Party", "New Open Non-Political Organised Leadership", "Newham Independents Party", "Official Monster Raving Loony Party", "One Leicester", "Party of Women", "People Before Profit Alliance", "Portsmouth Independent Party", "Propel", "Psychedelic Movement", "Putting Crewe First, Independent Residents Group", "Rebooting Democracy", "Rejoin EU", "Save Us Now", "Scottish Family Party", "Scottish Libertarian Party", "Scottish Socialist Party", "Shared Ground", "Social Democratic Party", "Social Justice party", "Socialist Equality Party", "Socialist Labour Party", "Sovereignty", "South Devon Alliance", "Stockport Fights Austerity No To Cuts", "Swale Independents", "Taking The Initiative Party", "The Common People", "The Mitre TW9", "The North East Party", "The Peace Party - Non-violence, Justice, Environment", "The Socialist Party of Great Britain", "The Yorkshire Party", "The Yoruba Party in the UK", "Trade Unionist and Socialist Coalition", "Transform Party", "True & Fair Party", "UK Voice", "Volt United Kingdom", "Women's Equality Party", "Workers Party of Britain", "Workers Revolutionary Party"],
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


def UNS(prev_election_dataset, forcast_dataset):
    prev_election_votes = get_party_votes(prev_election_dataset)
    forecast_votes = get_party_votes(forcast_dataset)
    prev_election_voteshare = get_voteshare(prev_election_votes)
    forecast_voteshare = get_voteshare(forecast_votes)
    voteshare_difference = {key: forecast_voteshare.get(key,0) -  prev_election_voteshare.get(key,0) for key in prev_election_voteshare.keys() | forecast_voteshare.keys()}
    indexes = get_index(prev_election_dataset[0])
    local_voteshare = np.concatenate([prev_election_dataset[1,index[0]:index[0] + index[1]] / sum(prev_election_dataset[1,index[0]:index[0] + index[1]]) for index in indexes])
    shifted_voteshare = [local_voteshare[x] + voteshare_difference[prev_election_dataset[2,x]] for x in range(len(prev_election_dataset[1]))]
    GE29_UNS = np.stack((prev_election_dataset[0],  shifted_voteshare, prev_election_dataset[2]))

    return get_seats(GE29_UNS)


def monte_carlo(model_outputs,iterations):
    mc_raw = []
    for i in range(iterations):
        votes = []
        for constituency in model_outputs:
            midpoints, standard_deviations = constituency[0], constituency[1]
            votes.append([standard_deviations[x] * np.random.randn() + midpoints[x] for x in range(len(midpoints))])
        mc_raw.append(get_seats(votes))
    return mc_raw




###-----------------------------------------Importing Data-----------------------------------------###
GE24 = pd.read_csv('candidate-level-results-general-election-04-07-2024.csv')
GE24_clean = np.stack((GE24['Constituency name'], GE24['Candidate vote count'] ,normalise_party_names(GE24['Main party name'])))

YouGov_all_adults = pd.read_excel('voting-intention (1).xlsx', sheet_name='All adults')
YouGov_party_names = normalise_party_names(YouGov_all_adults['Party'].iloc[:-2])
YouGov_all_adults_clean = np.stack((np.zeros(len(YouGov_party_names),),np.rint(YouGov_all_adults['2026-10-19'].iloc[:-2]*YouGov_all_adults['2026-10-19'].iloc[-2]),YouGov_party_names),dtype=object)


###-----------------------------------------Main-----------------------------------------###

print(UNS(GE24_clean, YouGov_all_adults_clean))
print(get_seats(GE24_clean))

#monte_carlo()















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

plot_parliament(UNS(GE24_clean, YouGov_all_adults_clean), baseline=get_seats(GE24_clean))

