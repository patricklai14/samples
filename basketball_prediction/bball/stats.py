import os

import numpy as np
import pandas as pd

from basketball_reference_web_scraper import client, data

from bball import constants, utils

import pdb

#fields in bball ref returned data
PLAYER_ID = "slug"
GAMES_PLAYED = "games_played"
FG_MADE = "made_field_goals"
THREES_MADE = "made_three_point_field_goals"
FT_MADE = "made_free_throws"
TEAM = "team"
LOCATION = "location"
OPPONENT = "opponent"

START_TIME = "start_time"
END_TIME = "end_time"
AWAY_TEAM = "away_team"
HOME_TEAM = "home_team"
AWAY_TEAM_SCORE = "away_team_score"
HOME_TEAM_SCORE = "home_team_score"

#constants
MIN_GAMES = 30
MIN_PPG = 10.
MAX_PPG = 25.

DATA_VERSION = "v2"

#TODO: make this a class to become more testable

#return the following statistics:
# - avg pts in previous season
# - avg pts 2 seasons ago
# - avg points in current season
# - avg points allowed by opposing team in current season
# - avg points allowed by opposing team to position
# - home game (1) or away game (0)
def write_data():
    for year in range(2002, 2020):
        year1 = year - 1
        year2 = year - 2

        player_stats_prev1_df = get_player_season_stats(year1)
        player_stats_prev2_df = get_player_season_stats(year2)
        #combine prev yrs into one df
        player_stats_prev_df = player_stats_prev1_df.merge(player_stats_prev2_df, left_index=True, right_index=True, 
                                                           suffixes=('_prev1', '_prev2'))
        eligible_players = player_stats_prev2_df.index

        #debug
        #eligible_players = ["butleji01", "townska01"]
        rolling_stats_df = get_rolling_player_season_stats(eligible_players, year)
        stats_df = rolling_stats_df.merge(player_stats_prev_df, left_on=constants.PLAYER_ID_COL, right_index=True)

        stats_df.to_csv("/home/patrick/data/bball/{}/data_{}.csv".format(DATA_VERSION, year))

def get_total_points(stats):
    return (2 * stats[FG_MADE]) + stats[THREES_MADE] + stats[FT_MADE]

#get player stats for a particular season
#exclude players who played less than MIN_GAMES
def get_player_season_stats(season):
    stats_by_player = {}

    all_player_stats = client.players_season_totals(season_end_year=season)
    players = []
    ppg = []
    teams = []

    for player_stats in all_player_stats:
        if player_stats[GAMES_PLAYED] < MIN_GAMES:
            continue

        total_points = get_total_points(player_stats)
        avg_points = float(total_points) / player_stats[GAMES_PLAYED]

        #filter players based on criteria
        if avg_points < MIN_PPG or avg_points > MAX_PPG:
            continue

        players.append(player_stats[PLAYER_ID])
        ppg.append(avg_points)
        teams.append(player_stats[TEAM])

    return pd.DataFrame({constants.PPG_COL: ppg}, index=players)

def get_reg_season_start_end_dates(season):
    schedule = client.season_schedule(season_end_year=season)

    start_date = utils.get_pacific_date(schedule[0][START_TIME])
    end_date = None

    games_played = 0
    for game in schedule:
        games_played += 1
        end_date = utils.get_pacific_date(game[START_TIME])

        if games_played == utils.get_num_games(season):
            break

    return start_date, end_date

#get stats for each game and rolling avgs at time of game for a player
#return a dataframe with the columns: [player, rolling_ppg_avg, pts],
#with each row representing a single game
def get_rolling_player_season_stats(players, season):
    schedule = client.season_schedule(season_end_year=season)
    start_date, end_date = get_reg_season_start_end_dates(season)

    HOME_GAME = 1
    AWAY_GAME = 0

    #TODO: move to own file
    class rolling_player_stats():
        def __init__(self):
            self.num_games = 0
            self.rolling_avg_pts = [-1.]
            self.game_pts = []
            self.home_not_away = []
            self.opp_allowed_rating = []

        def update_stats(self, points):
            self.game_pts.append(points)
            new_avg_pts = \
                (self.rolling_avg_pts[-1] * self.num_games + points) / (self.num_games + 1)
            self.rolling_avg_pts.append(new_avg_pts)
            self.num_games += 1

    #can be used for both individual and league average stats
    class rolling_team_stats():
        def __init__(self):
            self.num_games = 0
            self.avg_points_allowed = 0.

        def update_stats(self, points):
            self.avg_points_allowed = ((self.avg_points_allowed * self.num_games) + points) / (self.num_games + 1)
            self.num_games += 1


    player_stats_dict = {player: rolling_player_stats() for player in players}
    team_stats_dict = {}
    avg_team_stats = rolling_team_stats()

    game_number = 0
    for ts in pd.date_range(start=start_date, end=end_date, freq='D'):
        date = ts.date()
        print("Processing date: {}".format(date))

        #get all games for this date
        box_scores = client.player_box_scores(day= date.day, month=date.month, year=date.year)
        for player_box_score in box_scores:
            player = player_box_score[PLAYER_ID]

            if player not in player_stats_dict:
                continue

            #TODO: put calculations in rolling_player_stats
            player_stats = player_stats_dict[player]
            points = get_total_points(player_box_score)
            player_stats.home_not_away.append(
                HOME_GAME if player_box_score[LOCATION] == data.Location.HOME else AWAY_GAME)

            opposing_team = player_box_score[OPPONENT]
            if opposing_team in team_stats_dict:
                player_stats.opp_allowed_rating.append(
                    team_stats_dict[opposing_team].avg_points_allowed / avg_team_stats.avg_points_allowed)
            else:
                player_stats.opp_allowed_rating.append(-1.)

            player_stats.update_stats(points)

        #update team stats
        while utils.get_pacific_date(schedule[game_number][START_TIME]) <= date:
            game = schedule[game_number]
            away_team = game[AWAY_TEAM]
            home_team = game[HOME_TEAM]

            if away_team not in team_stats_dict:
                team_stats_dict[away_team] = rolling_team_stats()

            if home_team not in team_stats_dict:
                team_stats_dict[home_team] = rolling_team_stats()

            team_stats_dict[away_team].update_stats(game[HOME_TEAM_SCORE])
            team_stats_dict[home_team].update_stats(game[AWAY_TEAM_SCORE])

            #update league avg team stats
            avg_team_stats.update_stats(game[HOME_TEAM_SCORE])
            avg_team_stats.update_stats(game[AWAY_TEAM_SCORE])

            game_number += 1

    #convert to dataframe
    #Don't consider stats prior to this cutoff
    ROLLING_AVG_CUTOFF = 15

    points_data = []
    rolling_avg_pts_data = []
    opp_allowed_rating_data = []
    home_not_away_data = []
    player_id_data = []

    for player_id, player_stats in player_stats_dict.items():
        num_games = player_stats.num_games - ROLLING_AVG_CUTOFF
        if num_games < MIN_GAMES or player_stats.rolling_avg_pts[-1] < MIN_PPG:
            continue

        player_id_data += (num_games * [player_id])
        points_data += (player_stats.game_pts[ROLLING_AVG_CUTOFF:])
        opp_allowed_rating_data += (player_stats.opp_allowed_rating[ROLLING_AVG_CUTOFF:])
        home_not_away_data += (player_stats.home_not_away[ROLLING_AVG_CUTOFF:])

        #don't include last/final ppg
        rolling_avg_pts_data += (player_stats.rolling_avg_pts[ROLLING_AVG_CUTOFF:-1])

    return pd.DataFrame({constants.PLAYER_ID_COL: player_id_data, constants.PPG_COL: rolling_avg_pts_data, 
                         constants.PTS_COL: points_data, constants.OPP_ALLOWED_RATING_COL: opp_allowed_rating_data,
                         constants.HOME_NOT_AWAY_COL: home_not_away_data})

def get_rolling_team_stats(season):
    schedule = client.season_schedule(season_end_year=season)

    all_team_stats = {}
    games_played = 0
    for game in schedule:

        games_played += 1
        if games_played == utils.get_num_games(season):
            break

    return start_date, end_date

class data_wrapper:
    def __init__(self, data_df):
        self.data_df = data_df
        self.features = [constants.PPG_COL, constants.PPG_PREV1_COL, constants.PPG_PREV2_COL, 
                         constants.OPP_ALLOWED_RATING_COL, constants.HOME_NOT_AWAY_COL]

    def get_data(self):
        return self.data_df[self.features], self.data_df[constants.PTS_COL]

    def get_features(self):
        return self.features

def read_data(begin, end):
    DATA_DIR = "/home/patrick/data/bball/{}".format(DATA_VERSION)
    all_files = os.listdir(DATA_DIR)
    
    data_list = []
    columns = None
    for file in all_files:
        year = int(file[5:9])
        if year < begin or year > end:
            continue

        curr_data = pd.read_csv(os.path.join(DATA_DIR, file))
        data_list.append(curr_data.values)

        if columns is None:
            columns = curr_data.columns

    return data_wrapper(pd.DataFrame(np.vstack(data_list), columns=columns))


#get player stats for current season
def get_player_current_stats(players):
    CURRENT_SEASON = 2020
    season_stats_df = get_rolling_player_season_stats(players, CURRENT_SEASON)
    recent_stats_df = season_stats_df.groupby(constants.PLAYER_ID_COL).agg('last')

    return data_wrapper(recent_stats_df)