import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import random
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
np.random.seed(0)





party_naming_convention = {
    "Conservative": ["Conservative", "Con", "Cons", "Conservatives", "Conservative and Unionist", "Tory", "Tories", "con_share"],
    "Labour": ["Labour", "Lab", "Labour and Co-operative", "Labour Co-op", "Lab Co-op", "Lab/Co-op", "lab_share"],
    "Liberal Democrat": ["Liberal Democrat", "Lib Dem", "Lib Dems", "LibDem", "LibDems", "LD", "LDem", "Liberal Democrats","ld_share"],
    "Reform UK": ["Reform UK", "Reform", "Ref", "reform_share", "UK Independence Party", "UKIP", "United Kingdom Independence Party","The Brexit Party"],
    "Green Party": ["Green Party", "Green", "Greens", "Grn", "Green Party of England and Wales", "GPEW", "Scottish Green Party", "Scottish Greens", "Scottish Green", "SGP", "Green Party Northern Ireland", "Green Party NI", "green_share"],
    "Scottish National Party": ["Scottish National Party", "SNP", "Scottish Nationalist Party", "snp_pc_share"],
    "Plaid Cymru": ["Plaid Cymru", "Plaid", "PC", "Plaid Cymru - The Party of Wales", "Plaid Cymru - Party of Wales", "snp_pc_share"],
    "Other": ["nan" ,"other_share", "Other", "Others", "Oth", "Other parties", "Minor parties", "Independent", "Independents", "Ind", "IND", "The Workers' Party", "Open Borders Party", "Pirate Party UK", "Better for Bradford", "Money Free Party", "Concordia", "Humanity", "Wessex Regionalists", "Friends Party", "Scotland's Independence Referendum Party", "The Radical Party", "Something New", "Populist Party", "The Realists' Party", "The Just Political Party", "Apolitical Democrats", "North of England Community Alliance", "Independent Sovereign Democratic Britain", "Demos Direct Initiative Party", "The New Society of Worth", "Independent Save Withybush Save Lives", "Movement for Active Democracy (M.A.D.)", "Greater Manchester Homeless Voice", "Southampton Independents", "Southend Independent Association", "War Veteran's Pro-Traditional Family Party", "Citizens Independent Social Thought Alliance", "Compass Party", "National Front", "Justice for Men & Boys", "Vapers In Power", "Yorkshire First", "Patriotic Socialist Party", "All People's Party", "Left Unity", "The Republican Socialist Party", "Cannabis Is Safer Than Alcohol", "Red Flag - Anti-Corruption", "The 30-50 Coalition", "The Whig Party", "Respect Party", "Liberty GB", "The Northern Party", "Independence from Europe", "The Pilgrim Party", "Bournemouth Independent Alliance", "Independents for Bristol", "World Peace Through Song", "Alter Change - Politics. Only Different.", "The Above and Beyond Party", "Class War", "Christian Movement for Great Britain", "Mainstream", "The UK Progressive Democracy Party", "Putting Croydon First!", "British Independents", "Apni Party", "Europeans Party", "National Liberal Party - True Liberalism", "Beer, Baccy and Scratchings", "Guildford Greenbelt Group", "Campaign", "The U(niversal) Party", "Save Hartlepool Hospital", "Hoi Polloi", "Digital Democracy", "New Independent Centralists", "Democratic Reform Party", "The Pluralist Party", "People First - Gwerin Gyntaf", "The Sustainable Population Party", "Keep It Real Party", "Restore the Family For Children's Sake", "Party for a United Thanet", "The Party for Poole People Ltd.", "The Roman Party. Ave", "The Independent Political Alliance Party", "Islam Zinda Baad Platform", "Rochdale First Party", "Residents for Uttlesford", "We Are The Reality Party", "Children of the Atom", "The Democratic Party", "Al-Zebabist Nation of Ooog", "Manston Airport Independent Party", "Common Sense Party", "The Southport Party", "Ubuntu Party", "Population Party UK", "Free Public Transport Party", "The Principles of Politics Party", "Magna Carta Conservation Party Great Britain", "The Eccentric Party of Great Britain", "The Birthday Party", "Give Me Back Elmo", "Land Party", "Reduce VAT in Sport", "The Evolution Party", "Independent Kidderminster Hospital and Health Concern", "New Millennium Bean Party", "Scotland Against Crooked Lawyers", "English Independence Party", "The Scottish Jacobite Party", "Restoration Party", "All The South Party", "Hugh Salmon for Battersea Party", "A Vote Against MP Expense Abuse", "United Voice", "Trust", "Impact Party", "Local Liberals People Before Politics Party", "Bushra Irfan of Blackburn", "Integrity UK", "Blaenau Gwent People's Voice", "You Party", "The Science Party", "Scrap Members Allowances", "Democratic Nationalists", "Citizens for Undead Rights and Equality", "Bromsgrove Independent Conservative", "Cut The Deficit", "Alliance for Workers' Liberty", "Cambridge Socialists", "Get Snouts Out The Trough", "Money Reform Party", "Independent Save Our Green Belt", "The New Party", "Reduce Tax On Beer", "Blue Environment Party", "Tendring First", "Peoples Party Essex", "Socialist Alternative", "Justice Party", "Fancy Dress Party", "Independent Ealing Acton Communities Public Services", "Reform 2000 Party", "The Better Britain Party", "Best of a Bad Bunch", "Medway Independent Party", "The P.N.D.P.", "Matriarchal Party United Kingdom Great Britain", "The Magna Carta Party", "Direct Democracy (Communist) Party", "Independent Voice for Halifax", "Tamsin Omond to the Commons", "Your Right To Democracy Party Limited", "Animal Protection Party", "Save King George Hospital", "The Joy of Talk", "Middle England Party", "Unity for Peace and Socialism", "Independents Federation UK", "The Macclesfield Independent", "Freedom and Responsibility", "Mansfield Independents", "Solihull and Meriden Residents Association", "Nationwide Reform Party", "Northampton - Save Our Public Services", "Independents to Save Queen Mary's Hospital", "Equal Parenting Alliance", "Nobody Party", "The Youth Party", "Virtue Currency Cognitive Appraisal Party", "INDEPENDENT Leave-the-EU Alliance", "Fight For an Anti-War Government", "Campaign for Independent Politicians", "The Cornish Democrats", "No Candidate Deserves My Vote!", "City Independents", "Staffordshire Independent Group", "The True English (Poetry) Party", "Independent People Together", "Anticapitalists - Workers Power", "Democratic Labour Party", "Clause 28, Children's Protection Christian Democrats", "Democratic Unionist Party", "DUP","Sinn Féin", "Sinn Fein", "SF", "Democratic Unionists","Alba Party", "Alba", "Ulster Unionist Party", "UUP", "Ulster Unionists", "Social Democratic & Labour Party", "SDLP", "Alliance Party of Northern Ireland", "Alliance", "APNI", "Alliance NI", "Your Party", 'YP', "Aontú", "Aontu", "Traditional Unionist Voice", "TUV", "Speaker", "Spk", "Speaker seeking re-election", "TUSC", "OMRLP", "Monster Raving Loony", "CPA", "WPB", "SLP", "SSP", "WEP", "AWP", "PBP", "People Before Profit", "CPB", "SDP", "BNP", "British National Party", "Yorkshire", "Yorkshire Party", "English Democrats", "EDP", "Abolish the Welsh Assembly Party", "Alliance for Democracy and Freedom", "Alliance for Green Socialism", "Animal Welfare Party", "Ashfield Independents", "Blue Revolution", "British Democratic Party", "British Unionist Party B.U.P.", "Chesterfield And North Derbyshire Independents (CANDI)", "Christian Party, Proclaiming Christ's Lordship", "Christian Peoples Alliance", "Climate Party", "Common Good", "Communist Future", "Communist League Election Campaign", "Communist Party of Britain", "Confelicity", "Consensus", "Count Binface Party", "Cross-Community Labour Alternative", "Democracy for Chorley", "English Constitution Party", "Fairer Voting Party", "Freedom Alliance", "Hampshire Independents", "Heritage Party", "Independence for Scotland Party", "Independent Alliance (Kent)", "Independent Network", "Independent Oxford Alliance", "Independents for Direct Democracy", "Kingston Independent Residents Group", "Liberal", "Libertarian Party", "Lincolnshire Independents Lincolnshire First", "Liverpool Community Independents", "National Health Action Party", "New Open Non-Political Organised Leadership", "Newham Independents Party", "Official Monster Raving Loony Party", "One Leicester", "Party of Women", "People Before Profit Alliance", "Portsmouth Independent Party", "Propel", "Psychedelic Movement", "Putting Crewe First, Independent Residents Group", "Rebooting Democracy", "Rejoin EU", "Save Us Now", "Scottish Family Party", "Scottish Libertarian Party", "Scottish Socialist Party", "Shared Ground", "Social Democratic Party", "Social Justice party", "Socialist Equality Party", "Socialist Labour Party", "Sovereignty", "South Devon Alliance", "Stockport Fights Austerity No To Cuts", "Swale Independents", "Taking The Initiative Party", "The Common People", "The Mitre TW9", "The North East Party", "The Peace Party - Non-violence, Justice, Environment", "The Socialist Party of Great Britain", "The Yorkshire Party", "The Yoruba Party in the UK", "Trade Unionist and Socialist Coalition", "Transform Party", "True & Fair Party", "UK Voice", "Volt United Kingdom", "Women's Equality Party", "Workers Party of Britain", "Workers Revolutionary Party","Young People's Party","Veterans and People's Party","The Best for Luton Party", "Advance Together","Heavy Woollen District Independents", "Birkenhead Social Justice Party", "Yeshua","Space Navies Party", "The Universal Good Party", "Renew","Church of the Militant Elvis Party", "The Independent Group for Change","Burnley and Padiham Independent Party", "Gwlad Gwlad", "Patria", "Advance Together","The Cynon Valley Party", "Communities United Party","The Justice & Anti-Corruption Party", "Young People's Party","The Citizens Movement Party UK", "Veterans and People's Party","Touch Love Worldwide (UK)", "The Best for Luton Party", "Shropshire Party","Proud of Oldham & Saddleworth", "Cumbria First","Mebyon Kernow - The Party for Cornwall", "Motherworld Party","Wycombe Independents", "The Constitution and Reform Party"],
}

['London','Rest of South', 'Midlands', 'North', 'Scotland','Wales']

constituency_regions = {
    'London': ['Beckenham', 'Bethnal Green and Bow', 'Brent Central', 'Brent North', 'Bromley and Chislehurst', 'Camberwell and Peckham', 'Croydon Central', 'Croydon North', 'Ealing, Southall', 'Edmonton', 'Eltham', 'Enfield, Southgate', 'Hammersmith', 'Hampstead and Kilburn', 'Hornsey and Wood Green', 'Kensington', 'Lewisham West and Penge', 'Lewisham, Deptford', 'Streatham', 'Vauxhall', 'West Ham', 'Westminster North', 'Barking', 'Battersea', 'Beckenham and Penge', 'Bermondsey and Old Southwark', 'Bethnal Green and Stepney', 'Bexleyheath and Crayford', 'Brent East', 'Brent West', 'Brentford and Isleworth', 'Bromley and Biggin Hill', 'Carshalton and Wallington', 'Chelsea and Fulham', 'Chingford and Woodford Green', 'Chipping Barnet', 'Cities of London and Westminster', 'Clapham and Brixton Hill', 'Croydon East', 'Croydon South', 'Croydon West', 'Dagenham and Rainham', 'Dulwich and West Norwood', 'Ealing Central and Acton', 'Ealing North', 'Ealing Southall', 'East Ham', 'Edmonton and Winchmore Hill', 'Eltham and Chislehurst', 'Enfield North', 'Erith and Thamesmead', 'Feltham and Heston', 'Finchley and Golders Green', 'Greenwich and Woolwich', 'Hackney North and Stoke Newington', 'Hackney South and Shoreditch', 'Hammersmith and Chiswick', 'Hampstead and Highgate', 'Harrow East', 'Harrow West', 'Hayes and Harlington', 'Hendon', 'Holborn and St Pancras', 'Hornchurch and Upminster', 'Hornsey and Friern Barnet', 'Ilford North', 'Ilford South', 'Islington North', 'Islington South and Finsbury', 'Kensington and Bayswater', 'Kingston and Surbiton', 'Lewisham East', 'Lewisham North', 'Lewisham West and East Dulwich', 'Leyton and Wanstead', 'Mitcham and Morden', 'Old Bexley and Sidcup', 'Orpington', 'Peckham', 'Poplar and Limehouse', 'Putney', "Queen's Park and Maida Vale", 'Richmond Park', 'Romford', 'Ruislip, Northwood and Pinner', 'Southgate and Wood Green', 'Stratford and Bow', 'Streatham and Croydon North', 'Sutton and Cheam', 'Tooting', 'Tottenham', 'Twickenham', 'Uxbridge and South Ruislip', 'Vauxhall and Camberwell Green', 'Walthamstow', 'West Ham and Beckton', 'Wimbledon'],
    'Rest of South': ['Bridgwater and West Somerset', 'Brighton, Kemptown', 'Brighton, Pavilion', 'Bristol West', 'Broadland', 'Buckingham', 'Bury St Edmunds', 'Devizes', 'Dover', 'East Devon', 'Fareham', 'Henley', 'Hitchin and Harpenden', 'Hove', 'Isle of Wight', 'Kingswood', 'Luton South', 'Maidstone and The Weald', 'Meon Valley', 'Milton Keynes South', 'Mole Valley', 'North East Bedfordshire', 'North East Somerset', 'North Swindon', 'North Thanet', 'North Wiltshire', 'Plymouth, Moor View', 'Plymouth, Sutton and Devonport', 'Reading East', 'Reading West', 'Rochford and Southend East', 'Saffron Walden', 'Somerton and Frome', 'South East Cambridgeshire', 'South Swindon', 'South Thanet', 'South West Bedfordshire', 'South West Surrey', 'Southampton, Itchen', 'Southampton, Test', 'Southend West', 'Taunton Deane', 'The Cotswolds', 'Tiverton and Honiton', 'Tonbridge and Malling', 'Torridge and West Devon', 'Totnes', 'Wantage', 'Waveney', 'Wealden', 'Wells', 'Weston-Super-Mare', 'Aldershot', 'Arundel and South Downs', 'Ashford', 'Aylesbury', 'Banbury', 'Basildon and Billericay', 'Basingstoke', 'Bath', 'Beaconsfield', 'Bedford', 'Bexhill and Battle', 'Bicester and Woodstock', 'Bognor Regis and Littlehampton', 'Bournemouth East', 'Bournemouth West', 'Bracknell', 'Braintree', 'Brentwood and Ongar', 'Bridgwater', 'Brighton Kemptown and Peacehaven', 'Brighton Pavilion', 'Bristol Central', 'Bristol East', 'Bristol North East', 'Bristol North West', 'Bristol South', 'Broadland and Fakenham', 'Broxbourne', 'Buckingham and Bletchley', 'Bury St Edmunds and Stowmarket', 'Camborne and Redruth', 'Cambridge', 'Canterbury', 'Castle Point', 'Central Devon', 'Central Suffolk and North Ipswich', 'Chatham and Aylesford', 'Chelmsford', 'Cheltenham', 'Chesham and Amersham', 'Chichester', 'Chippenham', 'Christchurch', 'Clacton', 'Colchester', 'Crawley', 'Dartford', 'Didcot and Wantage', 'Dorking and Horley', 'Dover and Deal', 'Dunstable and Leighton Buzzard', 'Earley and Woodley', 'East Grinstead and Uckfield', 'East Hampshire', 'East Surrey', 'East Thanet', 'East Wiltshire', 'East Worthing and Shoreham', 'Eastbourne', 'Eastleigh', 'Ely and East Cambridgeshire', 'Epping Forest', 'Epsom and Ewell', 'Esher and Walton', 'Exeter', 'Exmouth and Exeter East', 'Fareham and Waterlooville', 'Farnham and Bordon', 'Faversham and Mid Kent', 'Filton and Bradley Stoke', 'Folkestone and Hythe', 'Forest of Dean', 'Frome and East Somerset', 'Gillingham and Rainham', 'Glastonbury and Somerton', 'Gloucester', 'Godalming and Ash', 'Gosport', 'Gravesham', 'Great Yarmouth', 'Guildford', 'Hamble Valley', 'Harlow', 'Harpenden and Berkhamsted', 'Harwich and North Essex', 'Hastings and Rye', 'Havant', 'Hemel Hempstead', 'Henley and Thame', 'Herne Bay and Sandwich', 'Hertford and Stortford', 'Hertsmere', 'Hitchin', 'Honiton and Sidmouth', 'Horsham', 'Hove and Portslade', 'Huntingdon', 'Ipswich', 'Isle of Wight East', 'Isle of Wight West', 'Lewes', 'Lowestoft', 'Luton North', 'Luton South and South Bedfordshire', 'Maidenhead', 'Maidstone and Malling', 'Maldon', 'Melksham and Devizes', 'Mid Bedfordshire', 'Mid Buckinghamshire', 'Mid Dorset and North Poole', 'Mid Norfolk', 'Mid Sussex', 'Milton Keynes Central', 'Milton Keynes North', 'New Forest East', 'New Forest West', 'Newbury', 'Newton Abbot', 'North Bedfordshire', 'North Cornwall', 'North Cotswolds', 'North Devon', 'North Dorset', 'North East Cambridgeshire', 'North East Hampshire', 'North East Hertfordshire', 'North East Somerset and Hanham', 'North Norfolk', 'North Somerset', 'North West Cambridgeshire', 'North West Essex', 'North West Hampshire', 'North West Norfolk', 'Norwich North', 'Norwich South', 'Oxford East', 'Oxford West and Abingdon', 'Peterborough', 'Plymouth Moor View', 'Plymouth Sutton and Devonport', 'Poole', 'Portsmouth North', 'Portsmouth South', 'Rayleigh and Wickford', 'Reading Central', 'Reading West and Mid Berkshire', 'Reigate', 'Rochester and Strood', 'Romsey and Southampton North', 'Runnymede and Weybridge', 'Salisbury', 'Sevenoaks', 'Sittingbourne and Sheppey', 'Slough', 'South Basildon and East Thurrock', 'South Cambridgeshire', 'South Cotswolds', 'South Devon', 'South Dorset', 'South East Cornwall', 'South Norfolk', 'South Suffolk', 'South West Devon', 'South West Hertfordshire', 'South West Norfolk', 'South West Wiltshire', 'Southampton Itchen', 'Southampton Test', 'Southend East and Rochford', 'Southend West and Leigh', 'Spelthorne', 'St Albans', 'St Austell and Newquay', 'St Ives', 'St Neots and Mid Cambridgeshire', 'Stevenage', 'Stroud', 'Suffolk Coastal', 'Surrey Heath', 'Sussex Weald', 'Swindon North', 'Swindon South', 'Taunton and Wellington', 'Tewkesbury', 'Thornbury and Yate', 'Thurrock', 'Tiverton and Minehead', 'Tonbridge', 'Torbay', 'Torridge and Tavistock', 'Truro and Falmouth', 'Tunbridge Wells', 'Watford', 'Waveney Valley', 'Weald of Kent', 'Wells and Mendip Hills', 'Welwyn Hatfield', 'West Dorset', 'West Suffolk', 'Weston-super-Mare', 'Winchester', 'Windsor', 'Witham', 'Witney', 'Woking', 'Wokingham', 'Worthing West', 'Wycombe', 'Yeovil'],
    'Midlands': ['Birmingham, Edgbaston', 'Birmingham, Erdington', 'Birmingham, Hall Green', 'Birmingham, Hodge Hill', 'Birmingham, Ladywood', 'Birmingham, Northfield', 'Birmingham, Perry Barr', 'Birmingham, Selly Oak', 'Birmingham, Yardley', 'Bosworth', 'Burton', 'Charnwood', 'Corby', 'Coventry North East', 'Dudley North', 'Dudley South', 'Grantham and Stamford', 'Halesowen and Rowley Regis', 'Harborough', 'Ludlow', 'Meriden', 'Mid Worcestershire', 'North Warwickshire', 'Nottingham North', 'Rutland and Melton', 'Sherwood', 'Shrewsbury and Atcham', 'Solihull', 'South Staffordshire', 'Stone', 'Walsall North', 'Walsall South', 'Warley', 'Wellingborough', 'West Bromwich East', 'West Bromwich West', 'Wolverhampton South West', 'Aldridge-Brownhills', 'Amber Valley', 'Ashfield', 'Bassetlaw', 'Birmingham Edgbaston', 'Birmingham Erdington', 'Birmingham Hall Green and Moseley', 'Birmingham Hodge Hill and Solihull North', 'Birmingham Ladywood', 'Birmingham Northfield', 'Birmingham Perry Barr', 'Birmingham Selly Oak', 'Birmingham Yardley', 'Bolsover', 'Boston and Skegness', 'Bromsgrove', 'Broxtowe', 'Burton and Uttoxeter', 'Cannock Chase', 'Chesterfield', 'Corby and East Northamptonshire', 'Coventry East', 'Coventry North West', 'Coventry South', 'Daventry', 'Derby North', 'Derby South', 'Derbyshire Dales', 'Droitwich and Evesham', 'Dudley', 'Erewash', 'Gainsborough', 'Gedling', 'Grantham and Bourne', 'Halesowen', 'Harborough, Oadby and Wigston', 'Hereford and South Herefordshire', 'High Peak', 'Hinckley and Bosworth', 'Kenilworth and Southam', 'Kettering', 'Kingswinford and South Staffordshire', 'Leicester East', 'Leicester South', 'Leicester West', 'Lichfield', 'Lincoln', 'Loughborough', 'Louth and Horncastle', 'Mansfield', 'Melton and Syston', 'Meriden and Solihull East', 'Mid Derbyshire', 'Mid Leicestershire', 'Newark', 'Newcastle-under-Lyme', 'North East Derbyshire', 'North Herefordshire', 'North Shropshire', 'North Warwickshire and Bedworth', 'North West Leicestershire', 'Northampton North', 'Northampton South', 'Nottingham East', 'Nottingham North and Kimberley', 'Nottingham South', 'Nuneaton', 'Redditch', 'Rugby', 'Rushcliffe', 'Rutland and Stamford', 'Sherwood Forest', 'Shrewsbury', 'Sleaford and North Hykeham', 'Smethwick', 'Solihull West and Shirley', 'South Derbyshire', 'South Holland and The Deepings', 'South Leicestershire', 'South Northamptonshire', 'South Shropshire', 'Stafford', 'Staffordshire Moorlands', 'Stoke-on-Trent Central', 'Stoke-on-Trent North', 'Stoke-on-Trent South', 'Stone, Great Wyrley and Penkridge', 'Stourbridge', 'Stratford-on-Avon', 'Sutton Coldfield', 'Tamworth', 'Telford', 'The Wrekin', 'Tipton and Wednesbury', 'Walsall and Bloxwich', 'Warwick and Leamington', 'Wellingborough and Rushden', 'West Bromwich', 'West Worcestershire', 'Wolverhampton North East', 'Wolverhampton South East', 'Wolverhampton West', 'Worcester', 'Wyre Forest'],
    'North': ['Barnsley Central', 'Barnsley East', 'Batley and Spen', 'Berwick-upon-Tweed', 'Blackley and Broughton', 'Blackpool North and Cleveleys', 'Blaydon', 'Blyth Valley', 'Bolton South East', 'Brigg and Goole', 'City of Chester', 'Cleethorpes', 'Copeland', 'Denton and Reddish', 'Dewsbury', 'Don Valley', 'East Yorkshire', 'Eddisbury', 'Ellesmere Port and Neston', 'Elmet and Rothwell', 'Garston and Halewood', 'Gateshead', 'Great Grimsby', 'Haltemprice and Howden', 'Halton', 'Hemsworth', 'Heywood and Middleton', 'Jarrow', 'Keighley', 'Kingston upon Hull North', 'Kingston upon Hull West and Hessle', 'Lancaster and Fleetwood', 'Leeds Central', 'Leeds West', 'Leigh', 'Liverpool, Riverside', 'Liverpool, Walton', 'Liverpool, Wavertree', 'Liverpool, West Derby', 'Manchester, Gorton', 'Manchester, Withington', 'Middlesbrough', 'Morley and Outwood', 'Newcastle upon Tyne Central', 'Newcastle upon Tyne East', 'Normanton, Pontefract and Castleford', 'North Tyneside', 'North West Durham', 'Oldham West and Royton', 'Pendle', 'Penrith and The Border', 'Pudsey', 'Richmond (Yorks)', 'Salford and Eccles', 'Sedgefield', 'Selby and Ainsty', 'Sheffield, Brightside and Hillsborough', 'Sheffield, Hallam', 'Sheffield, Heeley', 'Stockton South', 'Wakefield', 'Wansbeck', 'Washington and Sunderland West', 'Weaver Vale', 'Wentworth and Dearne', 'Wirral South', 'Workington', 'Worsley and Eccles South', 'Wyre and Preston North', 'Altrincham and Sale West', 'Ashton-under-Lyne', 'Barnsley North', 'Barnsley South', 'Barrow and Furness', 'Beverley and Holderness', 'Birkenhead', 'Bishop Auckland', 'Blackburn', 'Blackley and Middleton South', 'Blackpool North and Fleetwood', 'Blackpool South', 'Blaydon and Consett', 'Blyth and Ashington', 'Bolton North East', 'Bolton South and Walkden', 'Bolton West', 'Bootle', 'Bradford East', 'Bradford South', 'Bradford West', 'Bridlington and The Wolds', 'Brigg and Immingham', 'Burnley', 'Bury North', 'Bury South', 'Calder Valley', 'Carlisle', 'Cheadle', 'Chester North and Neston', 'Chester South and Eddisbury', 'Chorley', 'City of Durham', 'Colne Valley', 'Congleton', 'Cramlington and Killingworth', 'Crewe and Nantwich', 'Darlington', 'Dewsbury and Batley', 'Doncaster Central', 'Doncaster East and the Isle of Axholme', 'Doncaster North', 'Easington', 'Ellesmere Port and Bromborough', 'Fylde', 'Gateshead Central and Whickham', 'Goole and Pocklington', 'Gorton and Denton', 'Great Grimsby and Cleethorpes', 'Halifax', 'Harrogate and Knaresborough', 'Hartlepool', 'Hazel Grove', 'Hexham', 'Heywood and Middleton North', 'Houghton and Sunderland South', 'Huddersfield', 'Hyndburn', 'Jarrow and Gateshead East', 'Keighley and Ilkley', 'Kingston upon Hull East', 'Kingston upon Hull North and Cottingham', 'Kingston upon Hull West and Haltemprice', 'Knowsley', 'Lancaster and Wyre', 'Leeds Central and Headingley', 'Leeds East', 'Leeds North East', 'Leeds North West', 'Leeds South', 'Leeds South West and Morley', 'Leeds West and Pudsey', 'Leigh and Atherton', 'Liverpool Garston', 'Liverpool Riverside', 'Liverpool Walton', 'Liverpool Wavertree', 'Liverpool West Derby', 'Macclesfield', 'Makerfield', 'Manchester Central', 'Manchester Rusholme', 'Manchester Withington', 'Mid Cheshire', 'Middlesbrough South and East Cleveland', 'Middlesbrough and Thornaby East', 'Morecambe and Lunesdale', 'Newcastle upon Tyne Central and West', 'Newcastle upon Tyne East and Wallsend', 'Newcastle upon Tyne North', 'Newton Aycliffe and Spennymoor', 'Normanton and Hemsworth', 'North Durham', 'North Northumberland', 'Oldham East and Saddleworth', 'Oldham West, Chadderton and Royton', 'Ossett and Denby Dale', 'Pendle and Clitheroe', 'Penistone and Stocksbridge', 'Penrith and Solway', 'Pontefract, Castleford and Knottingley', 'Preston', 'Rawmarsh and Conisbrough', 'Redcar', 'Ribble Valley', 'Richmond and Northallerton', 'Rochdale', 'Rossendale and Darwen', 'Rother Valley', 'Rotherham', 'Runcorn and Helsby', 'Salford', 'Scarborough and Whitby', 'Scunthorpe', 'Sefton Central', 'Selby', 'Sheffield Brightside and Hillsborough', 'Sheffield Central', 'Sheffield Hallam', 'Sheffield Heeley', 'Sheffield South East', 'Shipley', 'Skipton and Ripon', 'South Ribble', 'South Shields', 'Southport', 'Spen Valley', 'St Helens North', 'St Helens South and Whiston', 'Stalybridge and Hyde', 'Stockport', 'Stockton North', 'Stockton West', 'Stretford and Urmston', 'Sunderland Central', 'Tatton', 'Thirsk and Malton', 'Tynemouth', 'Wakefield and Rothwell', 'Wallasey', 'Warrington North', 'Warrington South', 'Washington and Gateshead South', 'West Lancashire', 'Westmorland and Lonsdale', 'Wetherby and Easingwold', 'Whitehaven and Workington', 'Widnes and Halewood', 'Wigan', 'Wirral West', 'Worsley and Eccles', 'Wythenshawe and Sale East', 'York Central', 'York Outer'],
    'Scotland': ['Angus', 'Argyll and Bute', 'Banff and Buchan', 'Coatbridge, Chryston and Bellshill', 'Cumbernauld, Kilsyth and Kirkintilloch East', 'Dundee East', 'Dundee West', 'Dunfermline and West Fife', 'East Dunbartonshire', 'East Kilbride, Strathaven and Lesmahagow', 'East Lothian', 'Edinburgh East', 'Glasgow Central', 'Glasgow North West', 'Glenrothes', 'Gordon', 'Inverclyde', 'Inverness, Nairn, Badenoch and Strathspey', 'Kirkcaldy and Cowdenbeath', 'Lanark and Hamilton East', 'Linlithgow and East Falkirk', 'Moray', 'Motherwell and Wishaw', 'Ochil and South Perthshire', 'Perth and North Perthshire', 'Ross, Skye and Lochaber', 'Rutherglen and Hamilton West', 'Stirling', 'Aberdeen North', 'Aberdeen South', 'Aberdeenshire North and Moray East', 'Airdrie and Shotts', 'Alloa and Grangemouth', 'Angus and Perthshire Glens', 'Arbroath and Broughty Ferry', 'Argyll, Bute and South Lochaber', 'Ayr, Carrick and Cumnock', 'Bathgate and Linlithgow', 'Berwickshire, Roxburgh and Selkirk', 'Caithness, Sutherland and Easter Ross', 'Central Ayrshire', 'Coatbridge and Bellshill', 'Cowdenbeath and Kirkcaldy', 'Cumbernauld and Kirkintilloch', 'Dumfries and Galloway', 'Dumfriesshire, Clydesdale and Tweeddale', 'Dundee Central', 'Dunfermline and Dollar', 'East Kilbride and Strathaven', 'East Renfrewshire', 'Edinburgh East and Musselburgh', 'Edinburgh North and Leith', 'Edinburgh South', 'Edinburgh South West', 'Edinburgh West', 'Falkirk', 'Glasgow East', 'Glasgow North', 'Glasgow North East', 'Glasgow South', 'Glasgow South West', 'Glasgow West', 'Glenrothes and Mid Fife', 'Gordon and Buchan', 'Hamilton and Clyde Valley', 'Inverclyde and Renfrewshire West', 'Inverness, Skye and West Ross-shire', 'Kilmarnock and Loudoun', 'Livingston', 'Lothian East', 'Mid Dunbartonshire', 'Midlothian', 'Moray West, Nairn and Strathspey', 'Motherwell, Wishaw and Carluke', 'Na h-Eileanan an Iar', 'North Ayrshire and Arran', 'North East Fife', 'Orkney and Shetland', 'Paisley and Renfrewshire North', 'Paisley and Renfrewshire South', 'Perth and Kinross-shire', 'Rutherglen', 'Stirling and Strathallan', 'West Aberdeenshire and Kincardine', 'West Dunbartonshire'],
    'Wales': ['Aberavon', 'Aberconwy', 'Arfon', 'Blaenau Gwent', 'Brecon and Radnorshire', 'Cardiff Central', 'Carmarthen East and Dinefwr', 'Carmarthen West and South Pembrokeshire', 'Ceredigion', 'Clwyd South', 'Clwyd West', 'Cynon Valley', 'Delyn', 'Islwyn', 'Merthyr Tydfil and Rhymney', 'Monmouth', 'Montgomeryshire', 'Neath', 'Newport West', 'Ogmore', 'Preseli Pembrokeshire', 'Rhondda', 'Swansea East', 'Vale of Clwyd','Aberafan Maesteg', 'Alyn and Deeside', 'Bangor Aberconwy', 'Blaenau Gwent and Rhymney', 'Brecon, Radnor and Cwm Tawe', 'Bridgend', 'Caerfyrddin', 'Caerphilly', 'Cardiff East', 'Cardiff North', 'Cardiff South and Penarth', 'Cardiff West', 'Ceredigion Preseli', 'Clwyd East', 'Clwyd North', 'Dwyfor Meirionnydd', 'Gower', 'Llanelli', 'Merthyr Tydfil and Aberdare', 'Mid and South Pembrokeshire', 'Monmouthshire', 'Montgomeryshire and Glyndŵr', 'Neath and Swansea East', 'Newport East', 'Newport West and Islwyn', 'Pontypridd', 'Rhondda and Ogmore', 'Swansea West', 'Torfaen', 'Vale of Glamorgan', 'Wrexham', 'Ynys Môn'],
    'Northern Ireland': ['Belfast South','Belfast East', 'Belfast North', 'Belfast South and Mid Down', 'Belfast West', 'East Antrim', 'East Londonderry', 'Fermanagh and South Tyrone', 'Foyle', 'Lagan Valley', 'Mid Ulster', 'Newry and Armagh', 'North Antrim', 'North Down', 'South Antrim', 'South Down', 'Strangford', 'Upper Bann', 'West Tyrone']
}
ni_constituencies = constituency_regions['Northern Ireland']

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


def Uniform_Swing(prev_election_dataset, forcast_dataset,forecast_sample_size, raw=False, local_sd= 0.05, region="National"):
    prev_election_votes = get_party_votes(prev_election_dataset)
    forecast_votes = get_party_votes(forcast_dataset)
    prev_election_voteshare = get_voteshare(prev_election_votes)
    forecast_voteshare = get_voteshare(forecast_votes)
    voteshare_difference = {key: forecast_voteshare.get(key,0) -  prev_election_voteshare.get(key,0) for key in prev_election_voteshare.keys() | forecast_voteshare.keys()}
    indexes = get_index(prev_election_dataset[0])
    local_voteshare = np.concatenate([prev_election_dataset[1,index[0]:index[0] + index[1]] / sum(prev_election_dataset[1,index[0]:index[0] + index[1]]) for index in indexes])
    shifted_voteshare = np.clip(np.array([local_voteshare[x] + voteshare_difference[prev_election_dataset[2,x]] for x in range(len(prev_election_dataset[1]))]),a_min=0,a_max=None)
    sampling_error = {party: np.sqrt(voteshare *(1-voteshare)/ forecast_sample_size) for party,voteshare in forecast_voteshare.items()}      #sampling only, systematic error is added nationally in monte_carlo_raw
    if raw == True:
        shifted_sd = np.array([sampling_error.get(party,0) for party in prev_election_dataset[2]])
        stacked_output = [[prev_election_dataset[0, index[0]], [shifted_voteshare[index[0]:index[0] + index[1]], np.full(np.shape(shifted_voteshare[index[0]:index[0]+index[1]]), local_sd), shifted_sd[index[0]:index[0] + index[1]]], list(prev_election_dataset[2, index[0]:index[0] + index[1]]), region] for index in indexes]
        return stacked_output
    else:
        output = np.stack((prev_election_dataset[0], shifted_voteshare, prev_election_dataset[2]))
        return get_seats(output)


def monte_carlo_raw(model_outputs,iterations,systematic_sd=0.02):
    names = np.concatenate([[constituency[0]] * len(constituency[2]) for constituency in model_outputs])
    parties = np.concatenate([constituency[2] for constituency in model_outputs])
    regions = np.concatenate([[constituency[3] if len(constituency) > 3 else "National"] * len(constituency[2]) for constituency in model_outputs])
    means = np.concatenate([constituency[1][0] for constituency in model_outputs])
    local_sd = np.concatenate([constituency[1][1] for constituency in model_outputs])
    sampling_sd = np.concatenate([constituency[1][2] if len(constituency[1]) > 2 else np.zeros(len(constituency[2])) for constituency in model_outputs])
    party_list, party_index = np.unique(parties,return_inverse=True)
    region_list, region_index = np.unique(regions,return_inverse=True)

    # national (systematic) - one multiplier per party, applied to every seat the party stands in
    national_voteshare = get_voteshare(get_party_votes(np.stack((names, means, parties), dtype=object)))
    parties_voteshare = np.array([national_voteshare.get(party, 0) for party in party_list])
    seat_voteshare = np.bincount(party_index, weights=means) / np.maximum(np.bincount(party_index), 1)
    national_log_sd = np.minimum(systematic_sd / np.maximum(seat_voteshare, 1e-9), 0.3)

    # regional (sampling) - one shock per party per region
    pair_index = region_index * len(party_list) + party_index
    region_votes = np.bincount(pair_index, weights=means, minlength=len(region_list) * len(party_list)).reshape(len(region_list), len(party_list))
    regions_voteshare = region_votes / region_votes.sum(axis=1, keepdims=True)
    regions_sd = np.zeros(len(region_list) * len(party_list))
    regions_sd[pair_index] = sampling_sd
    regions_sd = regions_sd.reshape(len(region_list), len(party_list))
    regional_log_sd = np.minimum(regions_sd / np.maximum(regions_voteshare, 1e-9), 0.3)

    mc_raw = []
    for i in range(iterations):
        national_multiplier = np.exp(national_log_sd * np.random.randn(len(party_list)))
        national_multiplier = national_multiplier * parties_voteshare.sum() / (parties_voteshare * national_multiplier).sum()
        simulated_regional = regions_voteshare * np.exp(regional_log_sd * np.random.randn(len(region_list), len(party_list)))
        regional_shock_normalised = simulated_regional * regions_voteshare.sum(axis=1,keepdims=True) / simulated_regional.sum(axis=1, keepdims=True) - regions_voteshare
        voteshares = means * national_multiplier[party_index] + regional_shock_normalised[region_index, party_index] + (local_sd * np.random.randn(len(means)))
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


def random_forest(dataset):
    df = dataset
    target_cols = ['con_share', 'lab_share', 'ld_share', 'snp_pc_share', 'reform_share', 'green_share', 'other_share']
    y = df[target_cols]
    party_names_normalised = normalise_party_names(target_cols)

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

    # Time-aware split: hold out the most recent election (2024)
    holdout_year = df['year'].max()
    dev_mask = (df['year'] < holdout_year).to_numpy()
    X_dev, y_dev = X[dev_mask], y[dev_mask]
    X_test, y_test = X[~dev_mask], y[~dev_mask]

    # GridSearchCV with expanding-window folds over election years
    cv_splits = year_based_time_series_cv(df.loc[dev_mask, 'year'], n_splits=3)
    grid = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42, n_jobs=1),
        param_grid=PARAM_GRID,
        cv=cv_splits,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    grid.fit(X_dev, y_dev)
    print(f"[Pure RF] Best params: {grid.best_params_}")
    print(f"[Pure RF] Best CV MAE (time-series folds, raw %-point shares): {-grid.best_score_:.4f}")
    model = grid.best_estimator_

    # Evaluate on the held-out election (2024 backtest)
    raw_test_preds = model.predict(X_test)
    test_preds_df = pd.DataFrame(raw_test_preds, columns=target_cols, index=X_test.index)
    if 'reform_stood' in X_test.columns:
        test_preds_df['reform_share'] = test_preds_df['reform_share'] * df.loc[X_test.index, 'reform_stood']
    if 'green_stood' in X_test.columns:
        test_preds_df['green_share'] = test_preds_df['green_share'] * df.loc[X_test.index, 'green_stood']

    test_preds_normalized = test_preds_df.div(test_preds_df.sum(axis=1), axis=0)
    mae = mean_absolute_error(y_test / 100, test_preds_normalized)
    print(f"[Pure RF] Mean Absolute Error on {holdout_year} hold-out: {mae:.4f} (share, 0-1 scale)")

    backtest_shares = test_preds_normalized.to_numpy()
    backtest_names = (df.loc[X_test.index, 'name'] + df.loc[X_test.index, 'year'].astype(str)).to_numpy()
    backtest_output = [
        [backtest_names[x], [backtest_shares[x], np.full(len(target_cols), mae)], party_names_normalised]
        for x in range(len(backtest_shares))
    ]

    # Refit the tuned model on every election (including the hold-out year) for the 2026 forecast
    model = RandomForestRegressor(**grid.best_params_, random_state=42, n_jobs=-1)
    model.fit(X, y)

    future_df = pd.read_csv("prediction_dataset_new.csv")
    future_df['region_encoded'] = encoded_region.transform(future_df['region'])
    future_df['prev_winner_encoded'] = encoded_prev_winner.transform(future_df['prev_winner'])
    X_future = future_df[feature_cols].fillna(0)

    future_raw_preds = model.predict(X_future)
    future_preds_df = pd.DataFrame(future_raw_preds, columns=target_cols, index=future_df.index)
    future_preds_normalized = future_preds_df.div(future_preds_df.sum(axis=1), axis=0).to_numpy()

    constituency_names = future_df['name'] + future_df['year'].astype(str)
    output = [
        [constituency_names[x], [future_preds_normalized[x], np.full(np.shape(future_preds_normalized[x]), mae)], party_names_normalised]
        for x in range(len(future_preds_normalized))
    ]

    return backtest_output, output


def random_forest_UNS_hybrid(dataset):
    df = dataset
    target_cols = ['con_share', 'lab_share', 'ld_share', 'snp_pc_share', 'reform_share', 'green_share', 'other_share']
    prev_cols   = ['prev_con_share', 'prev_lab_share', 'prev_ld_share', 'prev_snp_pc_share',
                   'prev_reform_share', 'prev_green_share', 'prev_other_share']
    swing_cols  = ['nat_swing_con', 'nat_swing_lab', 'nat_swing_ld', 'nat_swing_snp_pc',
                   'nat_swing_reform', 'nat_swing_green', 'nat_swing_other']

    # 1. Linear Uniform National Swing (UNS) baseline (0-100 scale)
    y_raw = df[target_cols]
    y_baseline = pd.DataFrame(
        df[prev_cols].to_numpy() + df[swing_cols].to_numpy(),
        columns=target_cols,
        index=df.index
    )

    # 2. Target is local excess swing (actual share minus UNS baseline)
    y_residual = y_raw - y_baseline
    party_names_normalised = normalise_party_names(target_cols)

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

    # Time-aware split: hold out the most recent election (2024)
    holdout_year = df['year'].max()
    dev_mask = (df['year'] < holdout_year).to_numpy()
    X_dev, y_dev = X[dev_mask], y_residual[dev_mask]
    X_test = X[~dev_mask]
    y_raw_test = y_raw[~dev_mask]

    # GridSearchCV with expanding-window folds over election years
    cv_splits = year_based_time_series_cv(df.loc[dev_mask, 'year'], n_splits=3)
    grid = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42, n_jobs=1),
        param_grid=PARAM_GRID,
        cv=cv_splits,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    grid.fit(X_dev, y_dev)
    print(f"[Hybrid UNS-RF] Best params: {grid.best_params_}")
    print(f"[Hybrid UNS-RF] Best CV MAE (time-series folds, raw %-point shares): {-grid.best_score_:.4f}")
    model = grid.best_estimator_

    # 3. Reconstruct hold-out vote shares: UNS baseline + predicted local residual
    raw_test_preds = model.predict(X_test) + y_baseline.loc[X_test.index].to_numpy()
    test_preds_df = pd.DataFrame(raw_test_preds, columns=target_cols, index=X_test.index).clip(lower=0)

    if 'reform_stood' in df.columns:
        test_preds_df['reform_share'] = test_preds_df['reform_share'] * df.loc[X_test.index, 'reform_stood']
    if 'green_stood' in df.columns:
        test_preds_df['green_share'] = test_preds_df['green_share'] * df.loc[X_test.index, 'green_stood']

    test_preds_normalized = test_preds_df.div(test_preds_df.sum(axis=1), axis=0)

    # 4. Compare against original unshifted actual shares (y_raw_test / 100)
    mae = mean_absolute_error(y_raw_test / 100, test_preds_normalized)
    print(f"[Hybrid UNS-RF] Mean Absolute Error on {holdout_year} hold-out: {mae:.4f} (share, 0-1 scale)")

    backtest_shares = test_preds_normalized.to_numpy()
    backtest_names = (df.loc[X_test.index, 'name'] + df.loc[X_test.index, 'year'].astype(str)).to_numpy()
    backtest_output = [
        [backtest_names[x], [backtest_shares[x], np.full(len(target_cols), mae)], party_names_normalised]
        for x in range(len(backtest_shares))
    ]

    # 5. Refit on all historical residuals and reconstruct the 2026 forecast
    model = RandomForestRegressor(**grid.best_params_, random_state=42, n_jobs=-1)
    model.fit(X, y_residual)

    future_df = pd.read_csv("prediction_dataset_new.csv").fillna(0)
    future_df['region_encoded'] = encoded_region.transform(future_df['region'])
    future_df['prev_winner_encoded'] = encoded_prev_winner.transform(future_df['prev_winner'])
    X_future = future_df[feature_cols]

    future_baseline = future_df[prev_cols].to_numpy() + future_df[swing_cols].to_numpy()
    future_raw_preds = model.predict(X_future) + future_baseline
    future_preds_df = pd.DataFrame(future_raw_preds, columns=target_cols, index=future_df.index).clip(lower=0)

    if 'reform_stood' in future_df.columns:
        future_preds_df['reform_share'] = future_preds_df['reform_share'] * future_df['reform_stood']
    if 'green_stood' in future_df.columns:
        future_preds_df['green_share'] = future_preds_df['green_share'] * future_df['green_stood']

    future_preds_normalized = future_preds_df.div(future_preds_df.sum(axis=1), axis=0).to_numpy()
    constituency_names = (future_df['name'] + future_df['year'].astype(str)).to_numpy()
    output = [
        [constituency_names[x], [future_preds_normalized[x], np.full(len(target_cols), mae)], party_names_normalised]
        for x in range(len(future_preds_normalized))
    ]

    return backtest_output, output

def get_regions(dataset):
    constituency_to_region = {constituency: region for region, names in constituency_regions.items() for constituency in names}
    regions = {region: [] for region in constituency_regions}
    for i, constituency in enumerate(dataset[0]):
        try:
            regions[constituency_to_region[constituency]].append(i)
        except:
            print("Manual Review: ", constituency)
    return {region: dataset[:, index] for region, index in regions.items()}

def URS(prev_election,  forecast_regional_data, forecast_sample_sizes,regions, raw=False, local_sd=0.05):
    prev_election_regions = get_regions(prev_election)
    predicted_votes_by_region = {region: Uniform_Swing(prev_election_regions[region], forecast, sample_size, raw=raw, local_sd=local_sd, region=region) for region, forecast, sample_size in zip(regions, forecast_regional_data, forecast_sample_sizes)}
    if raw:
        return [constituency for output in predicted_votes_by_region.values() for constituency in output]
    else:
        totals = {}
        for region, (parties, counts) in predicted_votes_by_region.items():
            for party, count in zip(parties, counts):
                totals[party] = totals.get(party, 0) + count
        return list(totals.keys()), list(totals.values())



###-----------------------------------------Importing Data-----------------------------------------###
GE24 = pd.read_csv('candidate-level-results-general-election-04-07-2024.csv')
GE24_clean = np.stack((GE24['Constituency name'], GE24['Candidate vote count'] ,normalise_party_names(GE24['Main party name'])))
GE24_clean = GE24_clean[:, ~np.isin(GE24_clean[0], ni_constituencies)]

GE19 = pd.read_csv('HoC-GE2019-results-by-candidate.csv')
GE19_clean = np.stack((GE19['Constituency name'], GE19['Votes'] ,normalise_party_names(GE19['Party name'])))
GE19_clean = GE19_clean[:, ~np.isin(GE19_clean[0], ni_constituencies)]

GE17 = pd.read_csv('HoC-GE2017-results-by-candidate.csv')
GE17_clean = np.stack((GE17['Constituency name'], GE17['Votes'] ,normalise_party_names(GE17['Party name'])))
GE17_clean = GE17_clean[:, ~np.isin(GE17_clean[0], ni_constituencies)]

GE15 = pd.read_csv('HoC-GE2015-results-by-candidate.csv')
GE15_clean = np.stack((GE15['Constituency name'], GE15['Votes'] ,normalise_party_names(GE15['Party name'])))
GE15_clean = GE15_clean[:, ~np.isin(GE15_clean[0], ni_constituencies)]

GE10 = pd.read_csv('HoC-GE2010-results-by-candidate.csv')
GE10_clean = np.stack((GE10['Constituency name'], GE10['Votes'] ,normalise_party_names(GE10['Party name'])))
GE10_clean = GE10_clean[:, ~np.isin(GE10_clean[0], ni_constituencies)]

GE05 = pd.read_csv('candidate-level-results-notional-general-election-05-05-2005.csv')
GE05_clean = np.stack((GE05['Constituency name'], GE05['Candidate vote count'] ,normalise_party_names(GE05['Main party name'])))
GE05_clean = GE05_clean[:, ~np.isin(GE05_clean[0], ni_constituencies)]



YouGov_regions = ['London','Rest of South', 'Midlands', 'North', 'Scotland','Wales']

chosen_date = '2026-10-19'
YouGov_26_all_adults = pd.read_excel('voting-intention (1).xlsx', sheet_name='All adults')
YouGov_26_party_names = normalise_party_names(YouGov_26_all_adults['Party'].iloc[:-2])
YouGov_26_all_adults_clean = np.stack((np.zeros(len(YouGov_26_party_names),),np.rint(YouGov_26_all_adults[chosen_date].iloc[:-2]*YouGov_26_all_adults[chosen_date].iloc[-2]),YouGov_26_party_names),dtype=object)
YouGov_26_all_sample_size = YouGov_26_all_adults[chosen_date].iloc[-1]

YouGov_26_regional = [pd.read_excel('voting-intention (1).xlsx', sheet_name=region) for region in YouGov_regions]
YouGov_26_regional_clean = [np.stack((np.zeros(len(YouGov_26_party_names),),np.rint(YouGov_26_regional[x][chosen_date].iloc[:-2]*YouGov_26_regional[x][chosen_date].iloc[-2]),YouGov_26_party_names),dtype=object) for x in range(np.shape(YouGov_26_regional)[0])]
YouGov_26_regional_sample_sizes = [YouGov_26_regional[x][chosen_date].iloc[-1] for x in range(np.shape(YouGov_26_regional)[0])]

chosen_date = '2024-07-03'
YouGov_19_24_all_adults = pd.read_excel('voting-intention-2019-2024 (1).xlsx', sheet_name='All adults')
YouGov_19_24_party_names = normalise_party_names(YouGov_19_24_all_adults['If there were a general election held tomorrow, which party would you vote for?'].iloc[:-2])
YouGov_19_24_all_adults_clean = np.stack((np.zeros(len(YouGov_19_24_party_names),),np.rint(YouGov_19_24_all_adults[chosen_date].iloc[:-2]*YouGov_19_24_all_adults[chosen_date].iloc[-2]),YouGov_19_24_party_names),dtype=object)
YouGov_19_24_sample_size = YouGov_19_24_all_adults[chosen_date].iloc[-1]

YouGov_19_24_regional = [pd.read_excel('voting-intention-2019-2024 (1).xlsx', sheet_name=region) for region in YouGov_regions]
YouGov_19_24_regional_clean = [np.stack((np.zeros(len(YouGov_19_24_party_names),),np.rint(YouGov_19_24_regional[x][chosen_date].iloc[:-2]*YouGov_19_24_regional[x][chosen_date].iloc[-2]),YouGov_19_24_party_names),dtype=object) for x in range(np.shape(YouGov_19_24_regional)[0])]
YouGov_19_24_regional_sample_sizes = [YouGov_19_24_regional[x][chosen_date].iloc[-1] for x in range(np.shape(YouGov_19_24_regional)[0])]





YouGov_10_15_party_names = normalise_party_names(['Con', 'Lab', 'Lib Dem', 'UKIP', 'SNP', 'Green', 'Other'])
YouGov_10_15_wales_party_names = normalise_party_names(['Con', 'Lab', 'Lib Dem', 'UKIP', 'Plaid', 'Green', 'Other'])

YouGov_10_15_all_adults_clean = np.stack((np.zeros(7), np.rint(np.array([34, 34, 10, 12, 5, 4, 1]) / 100 * 10083), YouGov_10_15_party_names), dtype=object)
YouGov_10_15_sample_size = 10307

YouGov_10_15_regions = ['London', 'Rest of South', 'Midlands', 'North', 'Scotland', 'Wales']
YouGov_10_15_regional_shares = [[31, 42, 11, 9, 0, 5, 1], [43.1, 23.5, 12.4, 14.3, 0, 5.7, 0.7], [39, 35, 7, 14, 0, 4, 1],
                                [29, 44, 7, 12, 0, 5, 2], [14, 28, 7, 1, 48, 1, 1], [25, 39, 8, 12, 13, 2, 2]]
YouGov_10_15_regional_weighted_n = [1278, 3246, 1638, 2457, 949, 515]
YouGov_10_15_regional_sample_sizes = [1093, 2970, 1487, 2204, 1351, 1202]
YouGov_10_15_regional_clean = [np.stack((np.zeros(7), np.rint(np.array(s) / 100 * n), YouGov_10_15_wales_party_names if r == 'Wales' else YouGov_10_15_party_names), dtype=object)
                               for r, s, n in zip(YouGov_10_15_regions, YouGov_10_15_regional_shares, YouGov_10_15_regional_weighted_n)]




YouGov_15_17_party_names = normalise_party_names(['Con', 'Lab', 'Lib Dem', 'UKIP', 'SNP', 'Green', 'Other'])
YouGov_15_17_wales_party_names = normalise_party_names(['Con', 'Lab', 'Lib Dem', 'UKIP', 'Plaid', 'Green', 'Other'])

YouGov_15_17_all_adults_clean = np.stack((np.zeros(7), np.rint(np.array([42, 35, 10, 5, 5, 2, 1]) / 100 * 2130), YouGov_15_17_party_names), dtype=object)
YouGov_15_17_sample_size = 2130

YouGov_15_17_regions = ['London', 'Rest of South', 'Midlands', 'North', 'Scotland', 'Wales']
YouGov_15_17_regional_shares = [[37, 41, 12, 7, 0, 3, 0], [50, 30, 13, 4, 0, 2, 2], [43, 34, 10, 6, 0, 2, 2],
                                [38, 47, 7, 6, 0, 1, 1], [28, 22, 7, 0, 41, 0, 1], [43, 34, 10, 6, 3, 2, 2]]
YouGov_15_17_regional_weighted_n = [256, 707, 458, 513, 196, 458]
YouGov_15_17_regional_sample_sizes = [225, 747, 440, 524, 194, 440]
YouGov_15_17_regional_clean = [np.stack((np.zeros(7), np.rint(np.array(s) / 100 * n), YouGov_15_17_wales_party_names if r == 'Wales' else YouGov_15_17_party_names), dtype=object)
                               for r, s, n in zip(YouGov_15_17_regions, YouGov_15_17_regional_shares, YouGov_15_17_regional_weighted_n)]





YouGov_05_10_party_names = normalise_party_names(['Con', 'Lab', 'Lib Dem', 'UKIP', 'SNP', 'BNP', 'Green', 'Other'])
YouGov_05_10_wales_party_names = normalise_party_names(['Con', 'Lab', 'Lib Dem', 'UKIP', 'Plaid', 'BNP', 'Green', 'Other'])


YouGov_05_10_all_adults_clean = np.stack((np.zeros(8), np.rint(np.array([35, 28, 28, 3, 2, 2, 1, 1]) / 100 * 6483), YouGov_05_10_party_names), dtype=object)
YouGov_05_10_sample_size = 6483
YouGov_05_10_regions = ['London', 'Rest of South', 'Midlands', 'North', 'Scotland', 'Wales']
YouGov_05_10_regional_shares = [[35, 31, 26, 4, 0, 1, 1, 1], [43, 18, 31, 4, 0, 1, 1, 1], [37, 27, 27, 4, 0, 1, 1, 1],
                                [28, 38, 26, 3, 0, 3, 1, 1], [15, 37, 24, 1, 19, 1, 1, 1], [37, 27, 27, 4, 2, 1, 1, 1]]
YouGov_05_10_regional_weighted_n = [829, 2108, 1386, 1596, 564, 1386]
YouGov_05_10_regional_sample_sizes = [908, 2177, 1230, 1534, 634, 1230]
YouGov_05_10_regional_clean = [np.stack((np.zeros(8), np.rint(np.array(s) / 100 * n), YouGov_05_10_wales_party_names if r == 'Wales' else YouGov_05_10_party_names), dtype=object)
                               for r, s, n in zip(YouGov_05_10_regions, YouGov_05_10_regional_shares, YouGov_05_10_regional_weighted_n)]














collected_data = pd.read_csv("train_test_dataset_new.csv")                                       # Preparing the data into X feature matrix and y target vectors
collected_data = collected_data.fillna(0)                                                                # Filling in missing values
encoded_region = LabelEncoder()                                                  # Turning the string data to numerical values for the model to use
collected_data['region_encoded'] = encoded_region.fit_transform(collected_data['region'])
encoded_prev_winner = LabelEncoder()
collected_data['prev_winner_encoded'] = encoded_prev_winner.fit_transform(collected_data['prev_winner'])
###-----------------------------------------Main-----------------------------------------###

TESTS = []

# Run each model ONCE to get both 2024 backtest and 2026 forecast
RF_ALL_BACKTEST, RF_TEST_SET_2426 = random_forest(collected_data)
RF_TEST_SET_1924 = RF_ALL_BACKTEST[-631:]

HYBRID_ALL_BACKTEST, RF_UNS_HYBRID_TEST_SET_2426 = random_forest_UNS_hybrid(collected_data)
RF_UNS_HYBRID_TEST_SET_1924 = HYBRID_ALL_BACKTEST[-631:]
#--------------------------------------------2024 Backtest--------------------------------------
TESTS.append(monte_carlo_raw(RF_TEST_SET_1924, 1000))
TESTS.append(monte_carlo_raw(RF_UNS_HYBRID_TEST_SET_1924, 1000))

URS_TEST_SET_1924 = URS(GE19_clean, YouGov_19_24_regional_clean, YouGov_19_24_regional_sample_sizes, YouGov_regions, raw=True)
TESTS.append(monte_carlo_raw(URS_TEST_SET_1924, 1000))

UNS_TEST_SET_1924 = Uniform_Swing(GE19_clean, YouGov_19_24_all_adults_clean, YouGov_19_24_sample_size, raw=True)
TESTS.append(monte_carlo_raw(UNS_TEST_SET_1924, 1000))
#--------------------------------------------2026 Forecast--------------------------------------
TESTS.append(monte_carlo_raw(RF_TEST_SET_2426, 1000))
TESTS.append(monte_carlo_raw(RF_UNS_HYBRID_TEST_SET_2426, 1000))

URS_TEST_SET_2426 = URS(GE24_clean, YouGov_26_regional_clean, YouGov_26_regional_sample_sizes, YouGov_regions, raw=True)
TESTS.append(monte_carlo_raw(URS_TEST_SET_2426, 1000))

UNS_TEST_SET_2426 = Uniform_Swing(GE24_clean, YouGov_26_all_adults_clean, YouGov_26_all_sample_size, raw=True)
TESTS.append(monte_carlo_raw(UNS_TEST_SET_2426, 1000))
###-----------------------------------------Metrics (temporary AI-generated block)-----------------------------------------###

MODEL_NAMES = ["RF 2024 backtest", "RF_UNS_HYBRID 2024 backtest", "URS: 2019 result + YouGov Jul 2024 (backtest)", "UNS: 2019 result + YouGov Jul 2024 (backtest)", 
               "RF 2026 forecast (new data)","RF_UNS_HYBRID 2026 forecast (new data)", "URS: 2024 result + YouGov Oct 2026", "UNS: 2024 result + YouGov Oct 2026"]   # same order as TESTS
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




ORDER = ["Labour", "Conservative", "Liberal Democrat", "Reform UK",
         "Scottish National Party", "Others & NI", "Plaid Cymru", "Green Party"]
COLOURS = {"Labour": "#C8374A", "Conservative": "#4A84D4", "Liberal Democrat": "#E8A844",
           "Reform UK": "#6BB5CC", "Scottish National Party": "#CBA843",
           "Others & NI": "#9EA3AB", "Plaid Cymru": "#468B78", "Green Party": "#5BAA6A"}
MAJOR = {"Labour", "Conservative", "Liberal Democrat", "Reform UK", "Scottish National Party",
         "Plaid Cymru", "Green Party"}

# actual monte carlo graphs
mean = seats.mean(axis=0)
lo, hi = np.percentile(seats, [5, 95], axis=0)
order = [i for i in np.argsort(mean)[::-1] if hi[i] > 0]
names = [parties[i] for i in order]

def model_dict(test_set):
    model_mc_runs = monte_carlo_raw(test_set, 1000)
    model_parties_mc = sorted({p for u, _ in model_mc_runs for p in u})
    model_dicts = [dict(zip(u, c)) for u, c in model_mc_runs]
    model_seats_mc = np.array([[d.get(p, 0) for p in model_parties_mc] for d in model_dicts])

    mc_mean_dict = dict(zip(model_parties_mc, model_seats_mc.mean(axis=0)))
    mc_lo_dict = dict(zip(model_parties_mc, np.percentile(model_seats_mc, 5, axis=0)))
    mc_hi_dict = dict(zip(model_parties_mc, np.percentile(model_seats_mc, 95, axis=0)))
    return mc_mean_dict, mc_lo_dict, mc_hi_dict, model_mc_runs, model_parties_mc, model_seats_mc

# for 2026 data 

mc_mean_dict_uns26, mc_lo_dict_uns26, mc_hi_dict_uns26, uns_mc_runs26, uns_parties_mc26, uns_seats_mc26 = model_dict(UNS_TEST_SET_2426)
mc_mean_dict_urs26, mc_lo_dict_urs26, mc_hi_dict_urs26, urs_mc_runs26, urs_parties_mc26, urs_seats_mc26 = model_dict(URS_TEST_SET_2426)
mc_mean_dict_rf26,  mc_lo_dict_rf26,  mc_hi_dict_rf26, rf_runs_mc26, rf_parties_mc26, rf_seats_mc26 = model_dict(RF_TEST_SET_2426)
mc_mean_dict_hybrid26, mc_lo_dict_hybrid26, mc_hi_dict_hybrid26, hybrid_runs_mc26, hybrid_parties_mc26, hybrid_seats_mc26 = model_dict(RF_UNS_HYBRID_TEST_SET_2426)

# Get deterministic URS values 
det_parties_URS26, det_seats_URS26 = URS(GE24_clean, YouGov_26_regional_clean, YouGov_26_regional_sample_sizes, YouGov_regions, raw=False)
det_urs26 = dict(zip(det_parties_URS26, det_seats_URS26))
det_vals_URS26 = [det_urs26.get(p, 0) for p in names]

# Get deterministic UNS values 
det_parties_UNS26, det_seats_UNS26 = Uniform_Swing(GE24_clean, YouGov_26_all_adults_clean, YouGov_26_all_sample_size, raw=False)
det_uns26 = dict(zip(det_parties_UNS26, det_seats_UNS26))
det_vals_UNS26 = [det_uns26.get(p, 0) for p in names]

# seats, parties for random forest - deterministic
RF_raw26 = RF_TEST_SET_2426
rf_names26 = np.concatenate([[c[0]] * len(c[2]) for c in RF_raw26])
rf_shares26 = np.concatenate([c[1][0] for c in RF_raw26])
rf_parties26 = np.concatenate([c[2] for c in RF_raw26])

rf_dataset26 = np.stack((rf_names26, rf_shares26, rf_parties26), dtype=object)
rf_parties26, rf_seats26 = get_seats(rf_dataset26)

# seats, parties for random forest uns hybrid - deterministic
hybrid_raw26 = RF_UNS_HYBRID_TEST_SET_2426
hybrid_names26 = np.concatenate([[c[0]] * len(c[2]) for c in hybrid_raw26])
hybrid_shares26 = np.concatenate([c[1][0] for c in hybrid_raw26])
hybrid_parties26 = np.concatenate([c[2] for c in hybrid_raw26])

hybrid_dataset26 = np.stack((hybrid_names26, hybrid_shares26, hybrid_parties26), dtype=object)
hybrid_parties26, hybrid_seats26 = get_seats(hybrid_dataset26)

act_parties26, act_counts26 = get_seats(GE24_clean)
actual26 = dict(zip(act_parties26, act_counts26))
actual_vals26 = [actual26.get(p, 0) for p in names]  

# building chart 

all_parties = sorted(list(set(
    list(mc_mean_dict_uns26.keys()) + 
    list(mc_mean_dict_urs26.keys()) +
    list(mc_mean_dict_rf26.keys()) + 
    list(det_urs26.keys()) + 
    list(det_uns26.keys()) + 
    list(dict(zip(rf_parties26, rf_seats26)).keys()
))))

all_parties.sort(key=lambda p: mc_mean_dict_rf26.get(p, mc_mean_dict_uns26.get(p, 0)), reverse=True)

x = np.arange(len(all_parties))
w = 0.12

# UNS monte carlo
uns_mean26 = [mc_mean_dict_uns26.get(p, 0) for p in all_parties]
uns_lo26   = [mc_mean_dict_uns26.get(p, 0) - mc_lo_dict_uns26.get(p, 0) for p in all_parties]
uns_hi26   = [mc_hi_dict_uns26.get(p, 0) - mc_mean_dict_uns26.get(p, 0) for p in all_parties]

# URS monte carlo
urs_mean26 = [mc_mean_dict_urs26.get(p, 0) for p in all_parties]
urs_lo26   = [mc_mean_dict_urs26.get(p, 0) - mc_lo_dict_urs26.get(p, 0) for p in all_parties]
urs_hi26   = [mc_hi_dict_urs26.get(p, 0) - mc_mean_dict_urs26.get(p, 0) for p in all_parties]

# RF monte carlo 
rf_mean26 = [mc_mean_dict_rf26.get(p, 0) for p in all_parties]
rf_lo26   = [mc_mean_dict_rf26.get(p, 0) - mc_lo_dict_rf26.get(p, 0) for p in all_parties]
rf_hi26   = [mc_hi_dict_rf26.get(p, 0) - mc_mean_dict_rf26.get(p, 0) for p in all_parties]

# Hybrid monte carlo
hybrid_mean26 = [mc_mean_dict_hybrid26.get(p, 0) for p in all_parties]
hybrid_lo26   = [mc_mean_dict_hybrid26.get(p, 0) - mc_lo_dict_hybrid26.get(p, 0) for p in all_parties]
hybrid_hi26   = [mc_hi_dict_hybrid26.get(p, 0) - mc_mean_dict_hybrid26.get(p, 0) for p in all_parties]

rf_lo26 = np.clip(rf_lo26, 0, None)
rf_hi26 = np.clip(rf_hi26, 0, None)

uns_lo26 = np.clip(uns_lo26, 0, None )
uns_hi26 = np.clip(uns_hi26, 0, None )

urs_lo26 = np.clip(urs_lo26, 0, None )
urs_hi26 = np.clip(urs_hi26, 0, None )

hybrid_lo26 = np.clip(hybrid_lo26, 0, None)
hybrid_hi26 = np.clip(hybrid_hi26, 0, None)

# deterministic values
val_uns26 = [det_uns26.get(p, 0) for p in all_parties]
val_urs26 = [det_urs26.get(p, 0) for p in all_parties]
rf_det_dict26 = dict(zip(rf_parties26, rf_seats26))
val_rf_det26 = [rf_det_dict26.get(p, 0) for p in all_parties]
hybrid_det_dict26 = dict(zip(hybrid_parties26, hybrid_seats26))
val_hybrid_det26 = [hybrid_det_dict26.get(p, 0) for p in all_parties]

fig, ax = plt.subplots(figsize=(16, 6))

ax.bar(x - 3.5*w, hybrid_mean26, w, yerr=[hybrid_lo26, hybrid_hi26], capsize=2, color="#5E1468", label="Hybird (Monte Carlo Mean)")
ax.bar(x - 2.5*w, uns_mean26,   w, yerr=[uns_lo26, uns_hi26], capsize=2, color="#4A84D4", label="UNS (Monte Carlo Mean)")
ax.bar(x - 1.5*w, urs_mean26,   w, yerr=[urs_lo26, urs_hi26], capsize=2, color="#E8A844", label="URS (Monte Carlo Mean)")
ax.bar(x - 0.5*w, rf_mean26,    w, yerr=[rf_lo26, rf_hi26],   capsize=2, color="#2b5c8f", label="RF (Monte Carlo Mean)")
ax.bar(x + 0.5*w, val_hybrid_det26, w, color="#d9b0fb", label="Hybrid (Det)")
ax.bar(x + 1.5*w, val_uns26,    w, color="#C8374A", label="Plain UNS (Det)")
ax.bar(x + 2.5*w, val_urs26,    w, color="#f1c40f", label="Plain URS (Det)")
ax.bar(x + 3.5*w, val_rf_det26, w, color="#5BAA6A", label="Random Forest (Det)")
ax.scatter(names, actual_vals26, color="black", marker="*", s=150, zorder=4,
           label="Actual 2024 result")
ax.legend()
ax.axhline(326, color="black", linestyle="--", linewidth=1)
ax.text(len(all_parties) - 0.5, 330, "326 for a majority", ha="right", fontsize=8, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(all_parties, rotation=45, ha="right")
ax.set_xlabel("Party", fontweight="bold")
ax.set_ylabel("Predicted Seats", fontweight="bold")
ax.set_title("Comprehensive Comparison: Monte Carlo vs. Deterministic Models 2026", fontsize=13, fontweight="bold")

ax.legend(frameon=True, ncol=2)
plt.tight_layout()
plt.show()

# for 2024 data

mc_mean_dict_uns24, mc_lo_dict_uns24, mc_hi_dict_uns24, uns_mc_runs24, uns_parties_mc24, uns_seats_mc24 = model_dict(UNS_TEST_SET_1924)
mc_mean_dict_urs24, mc_lo_dict_urs24, mc_hi_dict_urs24, urs_mc_runs24, urs_parties_mc24, urs_seats_mc24 = model_dict(URS_TEST_SET_1924)
mc_mean_dict_rf24,  mc_lo_dict_rf24,  mc_hi_dict_rf24, rf_runs_mc24, rf_parties_mc24, rf_seats_mc24 = model_dict(RF_TEST_SET_1924)
mc_mean_dict_hybrid24, mc_lo_dict_hybrid24, mc_hi_dict_hybrid24, hybrid_runs_mc24, hybrid_parties_mc24, hybrid_seats_mc24 = model_dict(RF_UNS_HYBRID_TEST_SET_1924)

# Get deterministic URS values 
det_parties_URS24, det_seats_URS24 = URS(GE19_clean, YouGov_19_24_regional_clean, YouGov_19_24_regional_sample_sizes, YouGov_regions, raw=False)
det_urs24 = dict(zip(det_parties_URS24, det_seats_URS24))
det_vals_URS24 = [det_urs24.get(p, 0) for p in names]

# Get deterministic UNS values 
det_parties_UNS24, det_seats_UNS24 = Uniform_Swing(GE19_clean, YouGov_19_24_all_adults_clean, YouGov_19_24_sample_size, raw=False)
det_uns24 = dict(zip(det_parties_UNS24, det_seats_UNS24))
det_vals_UNS24 = [det_uns24.get(p, 0) for p in names]

# seats, parties for random forest - deterministic
RF_raw24 = RF_TEST_SET_1924
rf_names24 = np.concatenate([[c[0]] * len(c[2]) for c in RF_raw24])
rf_shares24 = np.concatenate([c[1][0] for c in RF_raw24])
rf_parties24 = np.concatenate([c[2] for c in RF_raw24])

rf_dataset24 = np.stack((rf_names24, rf_shares24, rf_parties24), dtype=object)
rf_parties24, rf_seats24 = get_seats(rf_dataset24)

# seats, parties for random forest uns hybrid - deterministic
hybrid_raw24 = RF_UNS_HYBRID_TEST_SET_1924
hybrid_names24 = np.concatenate([[c[0]] * len(c[2]) for c in hybrid_raw24])
hybrid_shares24 = np.concatenate([c[1][0] for c in hybrid_raw24])
hybrid_parties24 = np.concatenate([c[2] for c in hybrid_raw24])

hybrid_dataset24 = np.stack((hybrid_names24, hybrid_shares24, hybrid_parties24), dtype=object)
hybrid_parties24, hybrid_seats24 = get_seats(hybrid_dataset24)

act_parties24, act_counts24 = get_seats(GE19_clean)
actual24 = dict(zip(act_parties24, act_counts24))
actual_vals24 = [actual24.get(p, 0) for p in names]  

# building chart 

all_parties = sorted(list(set(
    list(mc_mean_dict_uns24.keys()) + 
    list(mc_mean_dict_urs24.keys()) +
    list(mc_mean_dict_rf24.keys()) + 
    list(det_urs24.keys()) + 
    list(det_uns24.keys()) + 
    list(dict(zip(rf_parties24, rf_seats24)).keys()
))))

all_parties.sort(key=lambda p: mc_mean_dict_rf24.get(p, mc_mean_dict_uns24.get(p, 0)), reverse=True)

x = np.arange(len(all_parties))
w = 0.12

# UNS monte carlo
uns_mean24 = [mc_mean_dict_uns24.get(p, 0) for p in all_parties]
uns_lo24   = [mc_mean_dict_uns24.get(p, 0) - mc_lo_dict_uns24.get(p, 0) for p in all_parties]
uns_hi24   = [mc_hi_dict_uns24.get(p, 0) - mc_mean_dict_uns24.get(p, 0) for p in all_parties]

# URS monte carlo
urs_mean24 = [mc_mean_dict_urs24.get(p, 0) for p in all_parties]
urs_lo24   = [mc_mean_dict_urs24.get(p, 0) - mc_lo_dict_urs24.get(p, 0) for p in all_parties]
urs_hi24  = [mc_hi_dict_urs24.get(p, 0) - mc_mean_dict_urs24.get(p, 0) for p in all_parties]

# RF monte carlo 
rf_mean24 = [mc_mean_dict_rf24.get(p, 0) for p in all_parties]
rf_lo24   = [mc_mean_dict_rf24.get(p, 0) - mc_lo_dict_rf24.get(p, 0) for p in all_parties]
rf_hi24   = [mc_hi_dict_rf24.get(p, 0) - mc_mean_dict_rf24.get(p, 0) for p in all_parties]

# Hybrid monte carlo
hybrid_mean24 = [mc_mean_dict_hybrid24.get(p, 0) for p in all_parties]
hybrid_lo24   = [mc_mean_dict_hybrid24.get(p, 0) - mc_lo_dict_hybrid24.get(p, 0) for p in all_parties]
hybrid_hi24   = [mc_hi_dict_hybrid24.get(p, 0) - mc_mean_dict_hybrid24.get(p, 0) for p in all_parties]

rf_lo24 = np.clip(rf_lo24, 0, None)
rf_hi24 = np.clip(rf_hi24, 0, None)

uns_lo24 = np.clip(uns_lo24, 0, None )
uns_hi24 = np.clip(uns_hi24, 0, None )

urs_lo24 = np.clip(urs_lo24, 0, None )
urs_hi24 = np.clip(urs_hi24, 0, None )

hybrid_lo24 = np.clip(hybrid_lo24, 0, None)
hybrid_hi24 = np.clip(hybrid_hi24, 0, None)

# deterministic values
val_uns24 = [det_uns24.get(p, 0) for p in all_parties]
val_urs24 = [det_urs24.get(p, 0) for p in all_parties]
rf_det_dict24 = dict(zip(rf_parties24, rf_seats24))
val_rf_det24 = [rf_det_dict24.get(p, 0) for p in all_parties]
hybrid_det_dict24 = dict(zip(hybrid_parties24, hybrid_seats24))
val_hybrid_det24 = [hybrid_det_dict24.get(p, 0) for p in all_parties]

fig, ax = plt.subplots(figsize=(16, 6))

ax.bar(x - 3.5*w, hybrid_mean24, w, yerr=[hybrid_lo24, hybrid_hi24], capsize=2, color="#5E1468", label="Hybird (Monte Carlo Mean)")
ax.bar(x - 2.5*w, uns_mean24,   w, yerr=[uns_lo24, uns_hi24], capsize=2, color="#4A84D4", label="UNS (Monte Carlo Mean)")
ax.bar(x - 1.5*w, urs_mean24,   w, yerr=[urs_lo24, urs_hi24], capsize=2, color="#E8A844", label="URS (Monte Carlo Mean)")
ax.bar(x - 0.5*w, rf_mean24,    w, yerr=[rf_lo24, rf_hi24],   capsize=2, color="#2b5c8f", label="RF (Monte Carlo Mean)")
ax.bar(x + 0.5*w, val_hybrid_det24, w, color="#d9b0fb", label="Hybrid (Det)")
ax.bar(x + 1.5*w, val_uns24,    w, color="#C8374A", label="Plain UNS (Det)")
ax.bar(x + 2.5*w, val_urs24,    w, color="#f1c40f", label="Plain URS (Det)")
ax.bar(x + 3.5*w, val_rf_det24, w, color="#5BAA6A", label="Random Forest (Det)")
ax.scatter(names, actual_vals24, color="black", marker="*", s=150, zorder=4,
           label="Actual 2019 result")
ax.legend()

ax.axhline(326, color="black", linestyle="--", linewidth=1)
ax.text(len(all_parties) - 0.5, 330, "326 for a majority", ha="right", fontsize=8, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(all_parties, rotation=45, ha="right")
ax.set_xlabel("Party", fontweight="bold")
ax.set_ylabel("Predicted Seats", fontweight="bold")
ax.set_title("Comprehensive Comparison: Monte Carlo vs. Deterministic Models 2024", fontsize=13, fontweight="bold")

ax.legend(frameon=True, ncol=2)
plt.tight_layout()
plt.show()

def coalition(parties_to_check, seats, parties):
    results = []
    for i, p in enumerate(parties_to_check):
        for q in parties_to_check[i+1:]:
            a, b = parties.index(p), parties.index(q)
            pair = (seats[:, a] + seats[:, b])
            need = (pair >= 326) & (seats[:, a]<326) & (seats[:,b]<326)
            prob = need.mean()
            if prob > 0.01:
                results.append((f"{p}+{q}", prob))
    results.sort(key=lambda r: r[1], reverse=True)
    return results

def get_winners(dataset):
    constituencies_raw, voteshare_raw, party_raw = dataset[0], dataset[1], dataset[2]
    constituencies_index = get_index(constituencies_raw)
    return [party_raw[np.argmax(voteshare_raw[x[0]:x[0]+x[1]])+x[0]] for x in constituencies_index]

def monte_carlo_winners(model_outputs,iterations):
    names = np.concatenate([[constituency[0]] * len(constituency[2]) for constituency in model_outputs])
    parties = np.concatenate([constituency[2] for constituency in model_outputs])

    all_winners = []
    for _ in range(iterations):
        votes = np.concatenate([constituency[1][1] * np.random.randn(len(constituency[1][0])) + constituency[1][0]
                                for constituency in model_outputs])
        all_winners.append(get_winners(np.stack((names, votes, parties), dtype=object)))
    return np.array(all_winners)

def marginal_seats(test_set):
    winners = monte_carlo_winners(test_set, 1000)
    seat_names = [c[0] for c in test_set]
    winner_parties = sorted(set(winners.flatten()))
    win_prob = np.array([[(winners[:, s] == p).mean() for p in winner_parties]
                     for s in range(winners.shape[1])])
    
    prob_const = pd.DataFrame(win_prob, index=seat_names, columns=winner_parties)
    most_likely = win_prob.max(axis=1)
    most_likely_party = [winner_parties[i] for i in win_prob.argmax(axis=1)]
    marginal = [(seat_names[s], most_likely_party[s], most_likely[s])
            for s in range(len(seat_names)) if most_likely[s] < 0.7]

    marginal_cons = np.array(seat_names)[most_likely < 0.5].tolist()

    margins = []
    for c in test_set:
        s = np.sort(c[1][0])[::-1]
        margins.append(s[0] - s[1] if len(s) > 1 else 1)
    margins = np.array(margins)

    return most_likely, marginal_cons, margins, marginal

# variables needed for each model in dict; test_set, seats
# e.g. random forest, RF_test_set, rf_mean_mc_seats 
TESTS_forstats = [UNS_TEST_SET_2426, URS_TEST_SET_2426, RF_TEST_SET_2426, RF_UNS_HYBRID_TEST_SET_2426, UNS_TEST_SET_1924, URS_TEST_SET_1924, RF_TEST_SET_1924, RF_UNS_HYBRID_TEST_SET_1924]
SEATS_forstats = [uns_seats_mc26, urs_seats_mc26, rf_seats_mc26, hybrid_seats_mc26, uns_seats_mc24, urs_seats_mc24, rf_seats_mc24, hybrid_seats_mc24]
PARTIES_forstats = [uns_parties_mc26, urs_parties_mc26, rf_parties_mc26, hybrid_parties_mc26, uns_seats_mc24, urs_seats_mc24, rf_seats_mc24, hybrid_seats_mc24]
MODELS_forstats = ['UNS26', 'URS26', 'RF26', 'HYBRID26', 'UNS24', 'URS24', 'RF24', 'HYBRID24']

for name, test_set, seats, parties in zip(MODELS_forstats, TESTS_forstats, SEATS_forstats, PARTIES_forstats):
    most_likely, marginal_cons, margins, marginal = marginal_seats(test_set)
    parties_to_check = [p for p in ORDER if p in parties]
    coalitions = coalition(parties_to_check, seats, parties)
    
    print(f"--- {name} ---")
    if coalitions:
        for pair_name, prob in coalitions:
            print(f"{pair_name}, {prob*100:.1f}%") 
        else: 
            print("No two-party coalition is needed in more than 1% of runs")
        
    print("Toss-ups (<50%):", (most_likely < 0.50).sum())
    print("Marginal (50-70%):", ((most_likely >= 0.50) & (most_likely < 0.70)).sum())
    print("Leaning (70-95%):", ((most_likely >= 0.70) & (most_likely < 0.95)).sum())
    print("Safe (>=95%):", (most_likely >= 0.95).sum())

    print((margins < 0.02).sum(), "seats with a lead under 2 points", name)
    print((margins < 0.05).sum(), "seats with a lead under 5 points", name)

    print('Toss up seats (0.5<):', marginal_cons, '\n') 

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

# plot_parliament(Uniform_Swing(GE24_clean, YouGov_26_all_adults_clean), baseline=get_seats(GE24_clean))

# distribution of number of seats for each party against runs x - (0-637), y (0-1000) - how many times the party achievied x amount of seats 

# add up for each party the runs of seats each run of monte carlo simulation and add it to a list 

# need to find what counts seats and records for each party 

# UNS 26 histogram attempt 

UNS_hist_array = uns_seats_mc26
UNS_hist_parties = uns_parties_mc26

plt.figure()
for party, data in zip(UNS_hist_parties, UNS_hist_array.T):
    bins = np.arange(data.min() - 0.5, data.max() + 1.5, 10)
    plt.hist(data, bins=bins, alpha=0.5, label=party, color=COLOURS.get(party, "#9EA3AB"))
plt.axvline(326, color="black", linestyle="--", linewidth=1)
plt.legend()
plt.xlabel("Seats")
plt.ylabel("Number of runs")
plt.title("UNS 26 histogram")
plt.show()

# URS 26 histogram
URS_hist_array = urs_seats_mc26
URS_hist_parties = urs_parties_mc26

plt.figure()
for party, data in zip(URS_hist_parties, URS_hist_array.T):
    bins = np.arange(data.min() - 0.5, data.max() + 1.5, 10)
    plt.hist(data, bins=bins, alpha=0.5, label=party, color=COLOURS.get(party, "#9EA3AB"))
plt.legend()
plt.xlabel("Seats")
plt.ylabel("Number of runs")
plt.title("URS 26 histogram")
plt.show()

# Random forest 26 histogram
RF_hist_array = rf_seats_mc26
RF_hist_parties = rf_parties_mc26

plt.figure()
for party, data in zip(RF_hist_parties, RF_hist_array.T):
    bins = np.arange(data.min() - 0.5, data.max() + 1.5, 10)
    plt.hist(data, bins=bins, alpha=0.5, label=party, color=COLOURS.get(party, "#9EA3AB"))
plt.legend()
plt.xlabel("Seats")
plt.ylabel("Number of runs")
plt.title("RF 26 histogram")
plt.show()

# Hybrid 26 histogram
hybrid_hist_array = hybrid_seats_mc26
hybrid_hist_parties = hybrid_parties_mc26

plt.figure()
for party, data in zip(hybrid_hist_parties, hybrid_hist_array.T):
    bins = np.arange(data.min() - 0.5, data.max() + 1.5, 10)
    plt.hist(data, bins=bins, alpha=0.5, label=party, color=COLOURS.get(party, "#9EA3AB"))
plt.legend()
plt.xlabel("Seats")
plt.ylabel("Number of runs")
plt.title("Hybrid 26 histogram")
plt.show()

# UNS 24 histogram attempt 
UNS_hist_array = uns_seats_mc24
UNS_hist_parties = uns_parties_mc24

plt.figure()
for party, data in zip(UNS_hist_parties, UNS_hist_array.T):
    bins = np.arange(data.min() - 0.5, data.max() + 1.5, 10)
    plt.hist(data, bins=bins, alpha=0.5, label=party, color=COLOURS.get(party, "#9EA3AB"))
plt.legend()
plt.xlabel("Seats")
plt.ylabel("Number of runs")
plt.title("UNS 24 histogram")
plt.show()

# URS histogram
URS_hist_array = urs_seats_mc24
URS_hist_parties = urs_parties_mc24

plt.figure()
for party, data in zip(URS_hist_parties, URS_hist_array.T):
    bins = np.arange(data.min() - 0.5, data.max() + 1.5, 10)
    plt.hist(data, bins=bins, alpha=0.5, label=party, color=COLOURS.get(party, "#9EA3AB"))
plt.legend()
plt.xlabel("Seats")
plt.ylabel("Number of runs")
plt.title("URS 24 histogram")
plt.show()

# Random forest histogram
RF_hist_array = rf_seats_mc24
RF_hist_parties =rf_parties_mc24

plt.figure()
for party, data in zip(RF_hist_parties, RF_hist_array.T):
    bins = np.arange(data.min() - 0.5, data.max() + 1.5, 10)
    plt.hist(data, bins=bins, alpha=0.5, label=party, color=COLOURS.get(party, "#9EA3AB"))
plt.legend()
plt.xlabel("Seats")
plt.ylabel("Number of runs")
plt.title("RF 24 histogram")
plt.show()

# Hybrid histogram
hybrid_hist_array = hybrid_seats_mc24
hybrid_hist_parties = hybrid_parties_mc24

plt.figure()
for party, data in zip(hybrid_hist_parties, hybrid_hist_array.T):
    bins = np.arange(data.min() - 0.5, data.max() + 1.5, 10)
    plt.hist(data, bins=bins, alpha=0.5, label=party, color=COLOURS.get(party, "#9EA3AB"))
plt.legend()
plt.xlabel("Seats")
plt.ylabel("Number of runs")
plt.title("Hybrid 24 histogram")
plt.show()
