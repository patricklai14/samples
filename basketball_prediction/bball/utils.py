import pytz

import pdb

#date utils
def get_pacific_date(time):
    return time.astimezone(pytz.timezone("America/Los_Angeles")).date()

def get_num_games(season):
    num_teams = 30 if season > 2004 else 29
    num_team_games = 66 if season == 2012 else 82

    return num_teams * num_team_games // 2 