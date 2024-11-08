# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 09:39:39 2024

@author: user
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import warnings

# 경고 무시 설정
warnings.filterwarnings('ignore')

result_df = pd.read_csv('datas/result_df.csv')
result_df_p = pd.read_csv('datas/result_df_p.csv')
df_swing_table_all = pd.read_csv('datas/df_swing_table_all.csv')

pd.set_option('display.max_columns', None)

import dash
import math
from dash import dcc, html, Input, Output, no_update
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial.distance import euclidean
from dash_table import DataTable

# Dash 애플리케이션 생성
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = ('Dashboard | Plate Discipline')
server = app.server

layer0_layout = html.Div([
    html.H1("Plate Discipline 유사도 프로그램", style={'margin-bottom': '20px'}),
    html.P("안녕하세요, Plate Discipline 유사도 프로그램을 방문해주셔서 감사합니다. "
            "해당 프로그램은 타자와 투수의 이름을 입력하면 유클리드 거리에 기반해서 입력한 선수와 유사한 선수들을 확인하실 수 있습니다.", style={'margin-bottom': '30px'}),
    
    html.H2("Plate Discipline", style={'margin-top': '40px', 'margin-bottom': '10px'}),
    html.P("타자가 타석에서 스트라이크존을 빠르게 파악하고, 침착하게 대응하는 능력을 뜻합니다. "
           "플레이트 디서플린은 타석에서 적응력과 참을성, 좋지 않은 공을 거를 수 있는 능력입니다. "
           "우열을 가리기보다는 타자의 캐릭터에 가까운 지표라고 볼 수 있습니다.", style={'margin-bottom': '30px'}),
    
    html.H2("유클리드 거리", style={'margin-top': '40px', 'margin-bottom': '10px'}),
    html.P("유클리드 거리는 두 점 사이의 직선 거리를 계산하는 방법으로, 두 점 간의 좌표 차이를 제곱하여 합산하고, 그 합의 제곱근을 구하는 방식으로 계산됩니다. "
           "이 프로그램에서는 선수의 Plate Discipline 지표들을 각각의 좌표로 간주하여 유클리드 거리를 계산하며, 이를 통해 두 선수의 유사성을 측정합니다. "
           "유클리드 거리가 짧을수록 두 선수의 Plate Discipline 특성이 유사하다고 해석할 수 있습니다.", style={'margin-bottom': '30px'}),
    
    html.H2("유사도 계산에 사용된 지표 - Plate Discipline", style={'margin-top': '40px', 'margin-bottom': '10px'}),
    
    DataTable(
        columns=[
            {"name": "지표", "id": "metric"},
            {"name": "계산식", "id": "formula"},
            {"name": "비고", "id": "note"},
            {"name": "타자 유사도 분석 포함 여부", "id": "batter_inclusion"},
            {"name": "투수 유사도 분석 포함 여부", "id": "pitcher_inclusion"}
        ],
        data=[
            {"metric": "O-Swing%", "formula": "존 밖의 투구 스윙 / 존 밖의 투구 수", "note": "", "batter_inclusion": "O", "pitcher_inclusion": "O"},
            {"metric": "Z-Swing%", "formula": "존 안의 투구 스윙 / 존 안의 투구 수", "note": "", "batter_inclusion": "O", "pitcher_inclusion": "O"},
            {"metric": "Swing%", "formula": "스윙 / 전체 투구", "note": "", "batter_inclusion": "O", "pitcher_inclusion": "O"},
            {"metric": "Zone%", "formula": "스트라이크 존 내의 투구 수 / 총 투구 수", "note": "유사도 분석에서는 포함하지 않음", "batter_inclusion": "X", "pitcher_inclusion": "X"},
            {"metric": "O-Contact%", "formula": "존 밖의 투구에 컨택한 투구 수 / 존 밖의 투구에 스윙한 투구 수", "note": "", "batter_inclusion": "O", "pitcher_inclusion": "O"},
            {"metric": "Z-Contact%", "formula": "존 안의 투구에 컨택한 투구 수 / 존 안의 투구에 스윙한 투구 수", "note": "", "batter_inclusion": "O", "pitcher_inclusion": "O"},
            {"metric": "Contact%", "formula": "컨택한 투구 수 / 스윙한 투구 수", "note": "", "batter_inclusion": "O", "pitcher_inclusion": "O"},
            {"metric": "First Pitch Swing%", "formula": "첫 번째 투구에 스윙한 투구 수 / 첫 번째 투구 수", "note": "타자 유사도 분석에서만 사용", "batter_inclusion": "O", "pitcher_inclusion": "X"},
            {"metric": "First Pitch Strike%", "formula": "첫 번째 투구가 스트라이크인 투구 수 / 첫 번째 투구 수", "note": "투수 유사도 분석에서만 사용", "batter_inclusion": "X", "pitcher_inclusion": "O"},
            {"metric": "SwStr%", "formula": "헛스윙 / 총 투구 수", "note": "", "batter_inclusion": "O", "pitcher_inclusion": "O"},
            {"metric": "Cstr%", "formula": "콜 스트라이크 / 총 투구 수", "note": "", "batter_inclusion": "O", "pitcher_inclusion": "O"},
            {"metric": "Csw%", "formula": "헛스윙+콜 스트라이크 / 총 투구 수", "note": "SwStr%과 Cstr%를 합치면 Csw%입니다.", "batter_inclusion": "O", "pitcher_inclusion": "O"}
        ],
        style_cell={'textAlign': 'center'},
        style_header={
            'fontWeight': 'bold',
            'backgroundColor': '#f2f2f2'
        },
        style_table={'width': '100%', 'margin': 'auto', 'margin-bottom': '30px'}
    ),
    
    dcc.Markdown(
        """
        데이터는 Baseball Savant의 PBP 데이터를 사용했습니다. 따라서 Baseball Savant에서 보는 선수의 데이터와 약간의 차이가 있을 수 있습니다. 
        PBP 데이터에는 자동고의4구가 없고, 존 설정이 어떤지 확인할 수 없기 때문입니다.

        스트라이크 존 설정은 [이 곳](https://tangotiger.net/strikezone/zone%20chart.png)을 참고했습니다.
        """,
        style={'margin-top': '30px', 'margin-bottom': '30px'}
    )
])

# 1번 프로그램 레이아웃 (타자 유사도 분석)
layer1_layout = html.Div([
    html.H1("타자"),
    dcc.Dropdown(
        id='batter-dropdown',
        options=[{'label': name, 'value': name} for name in result_df['batter_name_id']],
        style={'width': '100%'}
    ),
    html.Div(id='batter-output-container'),
    html.Div(id='batter-similarity-graph-container'),
    html.Div(id='batter-similarity-graph-batted-ball-container'),
    html.Div(id='batter-percentile-bar-chart-container'),
    html.Div(id='batter-percentile-bar-chart-batted-ball-container'),
    html.Div(id='batter-difference-from-avg-chart-container'),
    html.Div(id='batter-difference-from-avg-chart-batted-ball-container'),
    html.Div(id='batter-euclidean-distance-scatter-container'),  # Euclidean distance scatter container 추가
])

# 2번 프로그램 레이아웃 (투수 유사도 분석)
layer2_layout = html.Div([
    html.H1("투수"),
    dcc.Dropdown(
        id='pitcher-dropdown',
        options=[{'label': name, 'value': name} for name in result_df_p['pitcher_name_id']],
        style={'width': '100%'}
    ),
    html.Div(id='pitcher-output-container'),
    html.Div(id='pitcher-similarity-graph-container'),
    html.Div(id='pitcher-similarity-graph-batted-ball-container'),
    html.Div(id='pitcher-percentile-bar-chart-container'),
    html.Div(id='pitcher-percentile-bar-chart-batted-ball-container'),
    html.Div(id='pitcher-difference-from-avg-chart-container'),
    html.Div(id='pitcher-difference-from-avg-chart-batted-ball-container'),
    html.Div(id='pitcher-euclidean-distance-scatter-container'),  # Euclidean distance scatter container 추가
])

# 탭을 사용하여 세 개의 레이어를 구분
app.layout = html.Div([
    dcc.Tabs(id='tabs', value='layer0', children=[
        dcc.Tab(label='시작하기에 앞서', value='layer0'),
        dcc.Tab(label='타자', value='layer1'),
        dcc.Tab(label='투수', value='layer2'),
    ]),
    html.Div(id='tabs-content')
])

#-----------------------------------------------------------------------------------------------------------------------------
# 타자 Plate Discipline 유사도를 계산할 표준 정규화된 컬럼 리스트
columns_to_compare_batter = [
    'out_zone_swing%_std', 'zone_swing%_std', 'swing%_std',
    'out_zone_contact%_std', 'zone_contact%_std',
    'contact%_std', 'first_pitch_swing%_std',
    'swstr%_std', 'cstr%_std', 'csw%_std'
]

# 타자 Batted Ball Type 유사도를 계산할 표준 정규화된 컬럼 리스트
columns_to_compare_batter_batted_ball = [
    'exitspeed_avg_std', 'angle_avg_std', 'exitspeed_max_std', 'gb%_std', 'fb%_std', 'ld%_std', 'pu%_std'					
]

# 함께 출력할 원래 컬럼 리스트
original_columns_batter = [
    'out_zone_swing%', 'zone_swing%', 'swing%', 'zone%',
    'out_zone_contact%', 'zone_contact%', 'contact%',
    'first_pitch_strike%', 'first_pitch_swing%', 'swstr%',
    'cstr%', 'csw%', 'exitspeed_avg', 'angle_avg', 'exitspeed_max', 'gb%', 'fb%', 'ld%', 'pu%'						
]

# 추가할 타자 백분위수 컬럼 리스트
percentile_columns_batter = [f"{col}_percentile" for col in columns_to_compare_batter]

# 추가할 타자 백분위수 컬럼 리스트 (Batted Ball)
percentile_columns_batter_batted_ball = [f"{col}_percentile" for col in columns_to_compare_batter_batted_ball]

# 투수 Plate Discipline 유사도를 계산할 표준 정규화된 컬럼 리스트
columns_to_compare_pitcher = [
    'out_zone_swing%_std', 'zone_swing%_std', 'swing%_std',
    'out_zone_contact%_std', 'zone_contact%_std',
    'contact%_std', 'first_pitch_strike%_std',
    'swstr%_std', 'cstr%_std', 'csw%_std'
]

# 투수 Batted Ball Type 유사도를 계산할 표준 정규화된 컬럼 리스트
columns_to_compare_pitcher_batted_ball = [
    'exitspeed_avg_std', 'angle_avg_std', 'exitspeed_max_std', 'gb%_std', 'fb%_std', 'ld%_std', 'pu%_std'					
] 

# 함께 출력할 원래 컬럼 리스트
original_columns_pitcher = [
    'out_zone_swing%', 'zone_swing%', 'swing%', 'zone%',
    'out_zone_contact%', 'zone_contact%', 'contact%',
    'first_pitch_strike%', 'first_pitch_swing%', 'swstr%',
    'cstr%', 'csw%', 'exitspeed_avg', 'angle_avg', 'exitspeed_max', 'gb%', 'fb%', 'ld%', 'pu%'
]

# 추가할 투수 백분위수 컬럼 리스트
percentile_columns_pitcher = [f"{col}_percentile" for col in columns_to_compare_pitcher]

# 추가할 투수 백분위수 컬럼 리스트 (Batted Ball)
percentile_columns_pitcher_batted_ball = [f"{col}_percentile" for col in columns_to_compare_pitcher_batted_ball]

#-----------------------------------------------------------------------------------------------------------------------------

# 타자 평균과의 차이 계산 함수
def batter_calculate_difference_from_mean(row, df_swing_table_all):
    differences = {}
    player_game_year = row['game_year']

    # 해당 game_year에 맞는 평균값을 추출
    df_year = df_swing_table_all[df_swing_table_all['game_year'] == player_game_year].iloc[0]

    # 각 지표에 대해 평균과 차이를 계산
    for col in original_columns_batter:
        mean_value = df_year[col]
        differences[f"{col}_diff_from_avg"] = round(row[col] - mean_value, 2)

    return differences

# 투수 평균과의 차이 계산 함수
def pitcher_calculate_difference_from_mean(row, df_swing_table_all):
    differences = {}
    player_game_year = row['game_year']

    # 해당 game_year에 맞는 평균값을 추출
    df_year = df_swing_table_all[df_swing_table_all['game_year'] == player_game_year].iloc[0]

    # 각 지표에 대해 평균과 차이를 계산
    for col in original_columns_pitcher:
        mean_value = df_year[col]
        differences[f"{col}_diff_from_avg"] = round(row[col] - mean_value, 2)

    return differences

#-----------------------------------------------------------------------------------------------------------------------------

# 타자 Plate Discipline 유사도 계산 함수
def calculate_batter_similarity(batter_name_id_input):
    if batter_name_id_input not in result_df['batter_name_id'].values:
        print(f"Error: {batter_name_id_input} not found in the dataset.")
        return None

    target_row = result_df[result_df['batter_name_id'] == batter_name_id_input].iloc[0]
    target_batter = target_row['batter']
    target_pa = target_row['pa']

    df_exclude_self = result_df[result_df['batter'] != target_batter]
    target_values = target_row[columns_to_compare_batter].values

    df_exclude_self['euclidean_distance'] = df_exclude_self.apply(
        lambda row: euclidean(target_values, row[columns_to_compare_batter].values), axis=1
    )

    max_distance = df_exclude_self['euclidean_distance'].max()
    df_exclude_self['similarity_percent'] = round((1 - (df_exclude_self['euclidean_distance'] / max_distance)) * 100, 1)

    # df_swing_table_all에서 연도에 맞는 평균과의 차이를 계산
    differences_target = batter_calculate_difference_from_mean(target_row, df_swing_table_all)
    for key, value in differences_target.items():
        target_row[key] = value

    target_row_with_distance = target_row.copy()
    target_row_with_distance['euclidean_distance'] = 0.0
    target_row_with_distance['similarity_percent'] = 100.0

    df_with_diff = df_exclude_self.apply(lambda row: pd.Series(batter_calculate_difference_from_mean(row, df_swing_table_all)), axis=1)
    df_exclude_self = pd.concat([df_exclude_self, df_with_diff], axis=1).reset_index(drop=True)

    if 1 <= target_pa <= 100:
        result_df_pa = df_exclude_self[(df_exclude_self['pa'] >= 50) & (df_exclude_self['pa'] <= 100)]
    elif target_pa > 100:
        result_df_pa = df_exclude_self[(df_exclude_self['pa'] >= target_pa - 50) & (df_exclude_self['pa'] <= target_pa + 50)]
    else:
        print("Invalid pa range.")
        return None

    result_with_target = pd.concat([target_row_with_distance.to_frame().T, result_df_pa], ignore_index=True).reset_index(drop=True)

    return result_with_target[['batter_name_id', 'batter', 'batter_name_new', 'game_year', 'pa', 'euclidean_distance', 'similarity_percent']
                              + original_columns_batter + percentile_columns_batter + [f"{col}_diff_from_avg" for col in original_columns_batter]]

# 투수 Plate Discipline 유사도 계산 함수
def calculate_pitcher_similarity(pitcher_name_id_input):
    if pitcher_name_id_input not in result_df_p['pitcher_name_id'].values:
        print(f"Error: {pitcher_name_id_input} not found in the dataset.")
        return None

    target_row = result_df_p[result_df_p['pitcher_name_id'] == pitcher_name_id_input].iloc[0]
    target_pitcher = target_row['pitcher']
    target_pa = target_row['pa']

    df_exclude_self = result_df_p[result_df_p['pitcher'] != target_pitcher]
    target_values = target_row[columns_to_compare_pitcher].values

    df_exclude_self['euclidean_distance'] = df_exclude_self.apply(
        lambda row: euclidean(target_values, row[columns_to_compare_pitcher].values), axis=1
    )

    max_distance = df_exclude_self['euclidean_distance'].max()
    df_exclude_self['similarity_percent'] = round((1 - (df_exclude_self['euclidean_distance'] / max_distance)) * 100, 1)

    # df_swing_table_all에서 연도에 맞는 평균과의 차이를 계산
    differences_target = pitcher_calculate_difference_from_mean(target_row, df_swing_table_all)
    for key, value in differences_target.items():
        target_row[key] = value

    target_row_with_distance = target_row.copy()
    target_row_with_distance['euclidean_distance'] = 0.0
    target_row_with_distance['similarity_percent'] = 100.0

    df_with_diff = df_exclude_self.apply(lambda row: pd.Series(pitcher_calculate_difference_from_mean(row, df_swing_table_all)), axis=1)
    df_exclude_self = pd.concat([df_exclude_self, df_with_diff], axis=1).reset_index(drop=True)

    if 1 <= target_pa <= 100:
        result_df_pa = df_exclude_self[(df_exclude_self['pa'] >= 50) & (df_exclude_self['pa'] <= 100)]
    elif target_pa > 100:
        result_df_pa = df_exclude_self[(df_exclude_self['pa'] >= target_pa - 50) & (df_exclude_self['pa'] <= target_pa + 50)]
    else:
        print("Invalid pa range.")
        return None

    result_with_target = pd.concat([target_row_with_distance.to_frame().T, result_df_pa], ignore_index=True).reset_index(drop=True)

    return result_with_target[['pitcher_name_id', 'pitcher', 'pitcher_name_new', 'game_year', 'pa', 'euclidean_distance', 'similarity_percent']
                              + original_columns_pitcher + percentile_columns_pitcher + [f"{col}_diff_from_avg" for col in original_columns_pitcher]]

# 커스텀 빨강-파랑 색상 스케일 정의
custom_red_blue = [
    [0, 'blue'],   # 최소값은 파란색
    [0.5, 'white'], # 중앙값은 흰색
    [1, 'red']     # 최대값은 빨간색
]

#-----------------------------------------------------------------------------------------------------------------------------

# 타자 Batted Ball 유사도 계산 함수
def calculate_batter_similarity_batted_ball(batter_name_id_input):
    if batter_name_id_input not in result_df['batter_name_id'].values:
        print(f"Error: {batter_name_id_input} not found in the dataset.")
        return None

    target_row = result_df[result_df['batter_name_id'] == batter_name_id_input].iloc[0]
    target_batter = target_row['batter']
    target_pa = target_row['pa']

    df_exclude_self = result_df[result_df['batter'] != target_batter]
    target_values = target_row[columns_to_compare_batter_batted_ball].values

    df_exclude_self['euclidean_distance'] = df_exclude_self.apply(
        lambda row: euclidean(target_values, row[columns_to_compare_batter_batted_ball].values), axis=1
    )

    max_distance = df_exclude_self['euclidean_distance'].max()
    df_exclude_self['similarity_percent'] = round((1 - (df_exclude_self['euclidean_distance'] / max_distance)) * 100, 1)

    # df_swing_table_all에서 연도에 맞는 평균과의 차이를 계산
    differences_target = batter_calculate_difference_from_mean(target_row, df_swing_table_all)
    for key, value in differences_target.items():
        target_row[key] = value

    target_row_with_distance = target_row.copy()
    target_row_with_distance['euclidean_distance'] = 0.0
    target_row_with_distance['similarity_percent'] = 100.0

    df_with_diff = df_exclude_self.apply(lambda row: pd.Series(batter_calculate_difference_from_mean(row, df_swing_table_all)), axis=1)
    df_exclude_self = pd.concat([df_exclude_self, df_with_diff], axis=1).reset_index(drop=True)

    if 1 <= target_pa <= 100:
        result_df_pa = df_exclude_self[(df_exclude_self['pa'] >= 50) & (df_exclude_self['pa'] <= 100)]
    elif target_pa > 100:
        result_df_pa = df_exclude_self[(df_exclude_self['pa'] >= target_pa - 50) & (df_exclude_self['pa'] <= target_pa + 50)]
    else:
        print("Invalid pa range.")
        return None

    result_with_target = pd.concat([target_row_with_distance.to_frame().T, result_df_pa], ignore_index=True).reset_index(drop=True)

    return result_with_target[['batter_name_id', 'batter', 'batter_name_new', 'game_year', 'pa', 'euclidean_distance', 'similarity_percent']
                              + original_columns_batter + percentile_columns_batter_batted_ball + [f"{col}_diff_from_avg" for col in original_columns_batter]]

# 투수 Batted Ball 유사도 계산 함수
def calculate_pitcher_similarity_batted_ball(pitcher_name_id_input):
    if pitcher_name_id_input not in result_df_p['pitcher_name_id'].values:
        print(f"Error: {pitcher_name_id_input} not found in the dataset.")
        return None

    target_row = result_df_p[result_df_p['pitcher_name_id'] == pitcher_name_id_input].iloc[0]
    target_pitcher = target_row['pitcher']
    target_pa = target_row['pa']

    df_exclude_self = result_df_p[result_df_p['pitcher'] != target_pitcher]
    target_values = target_row[columns_to_compare_pitcher_batted_ball].values

    df_exclude_self['euclidean_distance'] = df_exclude_self.apply(
        lambda row: euclidean(target_values, row[columns_to_compare_pitcher_batted_ball].values), axis=1
    )

    max_distance = df_exclude_self['euclidean_distance'].max()
    df_exclude_self['similarity_percent'] = round((1 - (df_exclude_self['euclidean_distance'] / max_distance)) * 100, 1)

    # df_swing_table_all에서 연도에 맞는 평균과의 차이를 계산
    differences_target = pitcher_calculate_difference_from_mean(target_row, df_swing_table_all)
    for key, value in differences_target.items():
        target_row[key] = value

    target_row_with_distance = target_row.copy()
    target_row_with_distance['euclidean_distance'] = 0.0
    target_row_with_distance['similarity_percent'] = 100.0

    df_with_diff = df_exclude_self.apply(lambda row: pd.Series(pitcher_calculate_difference_from_mean(row, df_swing_table_all)), axis=1)
    df_exclude_self = pd.concat([df_exclude_self, df_with_diff], axis=1).reset_index(drop=True)

    if 1 <= target_pa <= 100:
        result_df_pa = df_exclude_self[(df_exclude_self['pa'] >= 50) & (df_exclude_self['pa'] <= 100)]
    elif target_pa > 100:
        result_df_pa = df_exclude_self[(df_exclude_self['pa'] >= target_pa - 50) & (df_exclude_self['pa'] <= target_pa + 50)]
    else:
        print("Invalid pa range.")
        return None

    result_with_target = pd.concat([target_row_with_distance.to_frame().T, result_df_pa], ignore_index=True).reset_index(drop=True)

    return result_with_target[['pitcher_name_id', 'pitcher', 'pitcher_name_new', 'game_year', 'pa', 'euclidean_distance', 'similarity_percent']
                              + original_columns_pitcher + percentile_columns_pitcher_batted_ball + [f"{col}_diff_from_avg" for col in original_columns_pitcher]]

#-----------------------------------------------------------------------------------------------------------------------------

# tabs-content를 업데이트하는 콜백 함수 추가
@app.callback(Output('tabs-content', 'children'), [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'layer0':
        return layer0_layout
    elif tab == 'layer1':
        return layer1_layout
    elif tab == 'layer2':
        return layer2_layout
    return no_update

# 타자
@app.callback(
    [Output('batter-output-container', 'children'),
     Output('batter-similarity-graph-container', 'children'),
     Output('batter-similarity-graph-batted-ball-container', 'children'),
     Output('batter-percentile-bar-chart-container', 'children'),
     Output('batter-percentile-bar-chart-batted-ball-container', 'children'),
     Output('batter-difference-from-avg-chart-container', 'children'),
     Output('batter-difference-from-avg-chart-batted-ball-container', 'children'),
     Output('batter-euclidean-distance-scatter-container', 'children')],
    [Input('batter-dropdown', 'value')]
)
def update_batter_output(batter_name_id_input):
    if batter_name_id_input is None:
        return "선수, 시즌을 선택해주세요 (MLB ID 6자리로도 검색 가능합니다)", no_update, no_update, no_update, no_update, no_update, no_update, no_update

    # Plate Discipline 유사도 분석 함수 호출
    similar_batters = calculate_batter_similarity(batter_name_id_input)

    # 결과가 없을 경우 처리
    if similar_batters is None or similar_batters.empty:
        return f"{batter_name_id_input} 선수의 데이터를 찾을 수 없습니다.", no_update, no_update, no_update, no_update, no_update, no_update, no_update

    # 결과를 필터링하여 Plate Discipline 유사도 상위 5명 추출
    similar_batters_top5 = similar_batters.sort_values(by='euclidean_distance').head(6)

    # 타자 Plate Discipline 유사도 테이블 생성
    similarity_table_batter = dcc.Graph(figure=display_table_for_batters(similar_batters_top5))

    # 타자 Batted Ball 유사도 분석 함수 호출
    similar_batters_bb = calculate_batter_similarity_batted_ball(batter_name_id_input)

    if similar_batters_bb is None or similar_batters_bb.empty:
        return f"{batter_name_id_input} 선수의 데이터를 찾을 수 없습니다.", no_update, no_update, no_update, no_update, no_update

    # 결과를 필터링하여 타자 Batted Ball 유사도 상위 5명 추출
    similar_batters_top5_bb = similar_batters_bb.sort_values(by='euclidean_distance').head(6)

    # Batted Ball 타자 유사도 테이블 생성
    similarity_table_batter_bb = dcc.Graph(figure=display_table_for_batters_batted_ball(similar_batters_top5_bb))
    
    # 타자 Plate Discipline 백분위수 그래프
    percentile_bar_chart_batter = dcc.Graph(figure=plot_percentile_bar_chart_for_batters(similar_batters_top5, batter_name_id_input))

    # 타자 Batted Ball 백분위수 그래프
    percentile_bar_chart_batter_batted_ball = dcc.Graph(figure=plot_percentile_bar_chart_for_batters_batted_ball(similar_batters_top5_bb, batter_name_id_input))

    # 타자 Plate Discipline 평균과의 차이 그래프
    first_batter_row = similar_batters_top5.iloc[0]
    batter_name = first_batter_row['batter_name_new']
    game_year_batter = int(first_batter_row['game_year'])
    difference_from_avg_chart_batter = dcc.Graph(figure=plot_difference_from_avg_for_batters(similar_batters_top5, df_swing_table_all, batter_name, game_year_batter))

    # 타자 Batted Ball 평균과의 차이 그래프
    first_batter_row_bb = similar_batters_top5_bb.iloc[0]
    batter_name_bb = first_batter_row_bb['batter_name_new']
    game_year_batter_bb = int(first_batter_row_bb['game_year'])
    difference_from_avg_chart_batter_batted_ball = dcc.Graph(figure=plot_difference_from_avg_for_batters_batted_ball(similar_batters_top5_bb, df_swing_table_all, batter_name_bb, game_year_batter_bb))

    # 유클리디안 거리 산포도
    euclidean_distance_scatter_batter = dcc.Graph(figure=plot_combined_euclidean_distance_scatter_batter(similar_batters, similar_batters_bb))

    # 출력 메시지
    output_message_batter = f"{game_year_batter} 시즌 {batter_name}의 유사도 분석 결과입니다."

    return output_message_batter, similarity_table_batter, percentile_bar_chart_batter, difference_from_avg_chart_batter, similarity_table_batter_bb, percentile_bar_chart_batter_batted_ball, difference_from_avg_chart_batter_batted_ball, euclidean_distance_scatter_batter

# 투수
@app.callback(
    [Output('pitcher-output-container', 'children'),
     Output('pitcher-similarity-graph-container', 'children'),
     Output('pitcher-similarity-graph-batted-ball-container', 'children'),
     Output('pitcher-percentile-bar-chart-container', 'children'),
     Output('pitcher-percentile-bar-chart-batted-ball-container', 'children'),
     Output('pitcher-difference-from-avg-chart-container', 'children'),
     Output('pitcher-difference-from-avg-chart-batted-ball-container', 'children'),
     Output('pitcher-euclidean-distance-scatter-container', 'children')],
    [Input('pitcher-dropdown', 'value')]
)
def update_pitcher_output(pitcher_name_id_input):
    if pitcher_name_id_input is None:
        return "선수, 시즌을 선택해주세요 (MLB ID 6자리로도 검색 가능합니다)", no_update, no_update, no_update, no_update, no_update, no_update, no_update

    # 투수 Plate Discipline 유사도 분석 함수 호출
    similar_pitchers = calculate_pitcher_similarity(pitcher_name_id_input)

    # 결과가 없을 경우 처리
    if similar_pitchers is None or similar_pitchers.empty:
        return f"선수 {pitcher_name_id_input}의 데이터를 찾을 수 없습니다.", no_update, no_update, no_update, no_update, no_update, no_update, no_update

    # 결과를 필터링하여 투수 Plate Discipline 유사도 상위 5명 추출
    similar_pitchers_top5 = similar_pitchers.sort_values(by='euclidean_distance', ascending=True).head(6)

    # 투수 Plate Discipline 유사도 테이블 생성
    similarity_table_pitcher = dcc.Graph(figure=display_table_for_pitchers(similar_pitchers_top5))

    # 투수 Batted Ball 유사도 분석 함수 호출
    similar_pitchers_bb = calculate_pitcher_similarity_batted_ball(pitcher_name_id_input)

    if similar_pitchers_bb is None or similar_pitchers_bb.empty:
        return f"{pitcher_name_id_input} 선수의 데이터를 찾을 수 없습니다.", no_update, no_update, no_update, no_update, no_update

    # 결과를 필터링하여 타자 Batted Ball 유사도 상위 5명 추출
    similar_pitchers_top5_bb = similar_pitchers_bb.sort_values(by='euclidean_distance').head(6)

    # Batted Ball 투수 유사도 테이블 생성
    similarity_table_pitcher_bb = dcc.Graph(figure=display_table_for_pitchers_batted_ball(similar_pitchers_top5_bb)) 

    # 투수 Plate Discipline 백분위수 그래프
    percentile_bar_chart_pitcher = dcc.Graph(figure=plot_percentile_bar_chart_for_pitchers(similar_pitchers_top5, pitcher_name_id_input))

    # 투수 Batted Ball 백분위수 그래프
    percentile_bar_chart_pitcher_batted_ball = dcc.Graph(figure=plot_percentile_bar_chart_for_pitchers_batted_ball(similar_pitchers_top5_bb, pitcher_name_id_input))

    # 투수 Plate Discipline 평균과의 차이 그래프
    first_pitcher_row = similar_pitchers_top5.iloc[0]
    pitcher_name = first_pitcher_row['pitcher_name_new']
    game_year_pitcher = int(first_pitcher_row['game_year'])
    difference_from_avg_chart_pitcher = dcc.Graph(figure=plot_difference_from_avg_for_pitchers(similar_pitchers_top5, df_swing_table_all, pitcher_name, game_year_pitcher))

    # 투수 Batted Ball 평균과의 차이 그래프
    first_pitcher_row_bb = similar_pitchers_top5_bb.iloc[0]
    pitcher_name_bb = first_pitcher_row_bb['pitcher_name_new']
    game_year_pitcher_bb = int(first_pitcher_row_bb['game_year'])
    difference_from_avg_chart_pitcher_batted_ball = dcc.Graph(figure=plot_difference_from_avg_for_pitchers_batted_ball(similar_pitchers_top5_bb, df_swing_table_all, pitcher_name_bb, game_year_pitcher_bb))
    
    # 유클리디안 거리 차트 생성
    euclidean_distance_scatter_pitcher = dcc.Graph(figure=plot_combined_euclidean_distance_scatter_pitcher(similar_pitchers, similar_pitchers_bb))

    # 결과 출력 메시지 생성
    output_message_pitcher = f"{game_year_pitcher} 시즌 {pitcher_name}의 유사도 분석 결과입니다."

    # 검색 후에만 그래프를 렌더링
    return output_message_pitcher, similarity_table_pitcher, percentile_bar_chart_pitcher, difference_from_avg_chart_pitcher, similarity_table_pitcher_bb, percentile_bar_chart_pitcher_batted_ball, difference_from_avg_chart_pitcher_batted_ball, euclidean_distance_scatter_pitcher

#-----------------------------------------------------------------------------------------------------------------------------

# Plate Discipline 유클리드 거리 상위 5명 추출 함수(타자)
def display_table_for_batters(similar_batters_top5):
    # 추출할 컬럼 리스트
    columns_to_display = [
        'game_year', 'batter_name_new', 'euclidean_distance', 'out_zone_swing%', 'zone_swing%', 'swing%', 'zone%',
        'out_zone_contact%', 'zone_contact%', 'contact%', 'first_pitch_swing%', 'swstr%', 'cstr%', 'csw%']

    # 'euclidean_distance' 컬럼을 소수점 셋째 자리까지 반올림
    similar_batters_top5['euclidean_distance'] = pd.to_numeric(similar_batters_top5['euclidean_distance'], errors='coerce').round(3)

    # 컬럼 이름 포맷팅
    formatted_columns = ['연도' if col == 'game_year' else
                         '타자' if col == 'batter_name_new' else
                         '유클라디안 거리' if col == 'euclidean_distance' else
                         col.replace('_', ' ').title() for col in columns_to_display]

    # 해당 컬럼들만 추출하여 plotly의 Table로 출력
    table_df = similar_batters_top5[columns_to_display]

    # 첫 번째 행의 game_year 값을 가져옴
    game_year = similar_batters_top5['game_year'].astype(int).iloc[0]
    batter_name = similar_batters_top5['batter_name_new'].iloc[0]

    # 리그 평균 추가
    league_avg_row = df_swing_table_all[df_swing_table_all['game_year'] == game_year]
    league_avg_row['batter_name_new'] = '리그 평균'
    league_avg_row['euclidean_distance'] = '-'
    league_avg_row = league_avg_row[columns_to_display]

    # 리그 평균을 기존 테이블의 맨 뒤에 추가
    table_df = pd.concat([table_df, league_avg_row], ignore_index=True)

    # Plotly 표 생성
    fig = go.Figure(data=[go.Table(
        header=dict(values=formatted_columns, fill_color='paleturquoise', align='center', height=40, font=dict(size=12)),
        cells=dict(values=[table_df[col] for col in table_df.columns], fill_color='white', align='center', height=40, font=dict(size=10))
    )])

    fig.update_layout(
        margin=dict(l=0, r=0, t=100, b=0),
        height=500,
        title=f"{game_year} Season {batter_name}<br>Top 5 Plate Discipline Similar Batters",
        title_x=0,
        title_xanchor='left',
        title_font=dict(size=20, family='Arial', color='black'))

    return fig

# 유클리드 거리 상위 5명 추출 함수(투수)
def display_table_for_pitchers(similar_pitchers_top5):
    # 추출할 컬럼 리스트
    columns_to_display = [
        'game_year', 'pitcher_name_new', 'euclidean_distance', 'out_zone_swing%', 'zone_swing%', 'swing%', 'zone%',
        'out_zone_contact%', 'zone_contact%', 'contact%', 'first_pitch_strike%', 'swstr%', 'cstr%', 'csw%']

    # 'euclidean_distance' 컬럼을 소수점 셋째 자리까지 반올림
    similar_pitchers_top5['euclidean_distance'] = pd.to_numeric(similar_pitchers_top5['euclidean_distance'], errors='coerce').round(3)

    # 컬럼 이름 포맷팅
    formatted_columns = ['연도' if col == 'game_year' else
                         '투수' if col == 'pitcher_name_new' else
                         '유클라디안 거리' if col == 'euclidean_distance' else
                         col.replace('_', ' ').title() for col in columns_to_display]

    # 해당 컬럼들만 추출하여 plotly의 Table로 출력
    table_df = similar_pitchers_top5[columns_to_display]

    # 첫 번째 행의 game_year 값을 가져옴
    game_year = similar_pitchers_top5['game_year'].astype(int).iloc[0]
    pitcher_name = similar_pitchers_top5['pitcher_name_new'].iloc[0]

    # 리그 평균 추가
    league_avg_row = df_swing_table_all[df_swing_table_all['game_year'] == game_year]
    league_avg_row['pitcher_name_new'] = '리그 평균'
    league_avg_row['euclidean_distance'] = '-'
    league_avg_row = league_avg_row[columns_to_display]

    # 리그 평균을 기존 테이블의 맨 뒤에 추가
    table_df = pd.concat([table_df, league_avg_row], ignore_index=True)

    # Plotly 표 생성
    fig = go.Figure(data=[go.Table(
        header=dict(values=formatted_columns, fill_color='paleturquoise', align='center', height=40, font=dict(size=12)),
        cells=dict(values=[table_df[col] for col in table_df.columns], fill_color='white', align='center', height=40, font=dict(size=10))
    )])

    fig.update_layout(
        margin=dict(l=0, r=0, t=100, b=0),
        height=500,
        title=f"{game_year} Season {pitcher_name}<br>Top 5 Plate Discipline Similar Pitchers",
        title_x=0,
        title_xanchor='left',
        title_font=dict(size=20, family='Arial', color='black'))

    return fig

#-----------------------------------------------------------------------------------------------------------------------------

# Batted Ball 유클리드 거리 상위 5명 추출 함수(타자)
def display_table_for_batters_batted_ball(similar_batters_top5_bb):
    # 추출할 컬럼 리스트
    columns_to_display = [
        'game_year', 'batter_name_new', 'euclidean_distance', 'exitspeed_avg', 'angle_avg', 'exitspeed_max', 'gb%', 'fb%', 'ld%', 'pu%']

    # 'euclidean_distance' 컬럼을 소수점 셋째 자리까지 반올림
    similar_batters_top5_bb['euclidean_distance'] = pd.to_numeric(similar_batters_top5_bb['euclidean_distance'], errors='coerce').round(3)

    # 컬럼 이름 포맷팅
    formatted_columns = ['연도' if col == 'game_year' else
                         '타자' if col == 'batter_name_new' else
                         '유클라디안 거리' if col == 'euclidean_distance' else
                         col.replace('_', ' ').title() for col in columns_to_display]

    # 해당 컬럼들만 추출하여 plotly의 Table로 출력
    table_df = similar_batters_top5_bb[columns_to_display]

    # 첫 번째 행의 game_year 값을 가져옴
    game_year = similar_batters_top5_bb['game_year'].astype(int).iloc[0]
    batter_name = similar_batters_top5_bb['batter_name_new'].iloc[0]

    # 리그 평균 추가
    league_avg_row = df_swing_table_all[df_swing_table_all['game_year'] == game_year]
    league_avg_row['batter_name_new'] = '리그 평균'
    league_avg_row['euclidean_distance'] = '-'
    league_avg_row = league_avg_row[columns_to_display]

    # 리그 평균을 기존 테이블의 맨 뒤에 추가
    table_df = pd.concat([table_df, league_avg_row], ignore_index=True)

    # Plotly 표 생성
    fig = go.Figure(data=[go.Table(
        header=dict(values=formatted_columns, fill_color='paleturquoise', align='center', height=40, font=dict(size=12)),
        cells=dict(values=[table_df[col] for col in table_df.columns], fill_color='white', align='center', height=40, font=dict(size=10))
    )])

    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        height=425,
        title=f"Top 5 Batted Ball Similar Batters",
        title_x=0,
        title_xanchor='left',
        title_font=dict(size=20, family='Arial', color='black'))

    return fig

# Batted Ball 유클리드 거리 상위 5명 추출 함수(투수)
def display_table_for_pitchers_batted_ball(similar_pitchers_top5_bb):
    # 추출할 컬럼 리스트
    columns_to_display = [
        'game_year', 'pitcher_name_new', 'euclidean_distance', 'exitspeed_avg', 'angle_avg', 'exitspeed_max', 'gb%', 'fb%', 'ld%', 'pu%']

    # 'euclidean_distance' 컬럼을 소수점 셋째 자리까지 반올림
    similar_pitchers_top5_bb['euclidean_distance'] = pd.to_numeric(similar_pitchers_top5_bb['euclidean_distance'], errors='coerce').round(3)

    # 컬럼 이름 포맷팅
    formatted_columns = ['연도' if col == 'game_year' else
                         '투수' if col == 'pitcher_name_new' else
                         '유클라디안 거리' if col == 'euclidean_distance' else
                         col.replace('_', ' ').title() for col in columns_to_display]

    # 해당 컬럼들만 추출하여 plotly의 Table로 출력
    table_df = similar_pitchers_top5_bb[columns_to_display]

    # 첫 번째 행의 game_year 값을 가져옴
    game_year = similar_pitchers_top5_bb['game_year'].astype(int).iloc[0]
    pitcher_name = similar_pitchers_top5_bb['pitcher_name_new'].iloc[0]

    # 리그 평균 추가
    league_avg_row = df_swing_table_all[df_swing_table_all['game_year'] == game_year]
    league_avg_row['pitcher_name_new'] = '리그 평균'
    league_avg_row['euclidean_distance'] = '-'
    league_avg_row = league_avg_row[columns_to_display]

    # 리그 평균을 기존 테이블의 맨 뒤에 추가
    table_df = pd.concat([table_df, league_avg_row], ignore_index=True)

    # Plotly 표 생성
    fig = go.Figure(data=[go.Table(
        header=dict(values=formatted_columns, fill_color='paleturquoise', align='center', height=40, font=dict(size=12)),
        cells=dict(values=[table_df[col] for col in table_df.columns], fill_color='white', align='center', height=40, font=dict(size=10))
    )])

    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        height=425,
        title=f"Top 5 Batted Ball Similar Pitchers",
        title_x=0,
        title_xanchor='left',
        title_font=dict(size=20, family='Arial', color='black'))

    return fig

#-----------------------------------------------------------------------------------------------------------------------------

def plot_percentile_bar_chart_for_batters(similar_batters, batter_name_id_input):
    # 6개의 타자 Plate Discipline 백분위수 컬럼 리스트
    batter_percentile_columns = [
        'contact%_std_percentile', 'zone_contact%_std_percentile',
        'zone_swing%_std_percentile', 'out_zone_swing%_std_percentile', 'swstr%_std_percentile', 'csw%_std_percentile']

    batter_percentile_columns_adj = [
        'contact%_std_adj_percentile', 'zone_contact%_std_adj_percentile',
        'zone_swing%_std_adj_percentile', 'out_zone_swing%_std_adj_percentile', 'swstr%_std_adj_percentile', 'csw%_std_adj_percentile']

    # target_adj를 사용하여 batter_name_id_input의 정보를 가져옴
    target_adj = result_df[result_df['batter_name_id'] == batter_name_id_input].iloc[0]
    
    # 선수의 PA와 game_year를 가져옴
    target_pa = similar_batters.iloc[0]['pa']
    game_year = similar_batters.iloc[0]['game_year']

    # game_year가 문자열인 경우 정수로 변환
    if isinstance(game_year, str):
        game_year = int(game_year)

    # game_year에 따라 pa_threshold 설정
    if game_year == 2020:
        pa_threshold = 126
    else:
        pa_threshold = 340

    # 선수의 실제 측정값 추출
    actual_columns = ['contact%', 'zone_contact%', 'zone_swing%', 'out_zone_swing%', 'swstr%', 'csw%']
    actual_values = similar_batters.iloc[0][actual_columns].values

    # PA 기준에 따라 백분위수 컬럼 선택
    if target_pa >= pa_threshold:
        percentile_values = target_adj[batter_percentile_columns_adj].astype(float).values  # 문자열을 실수형으로 변환
        text_values = [f"{int(float(value))}" for value in percentile_values]  # 소수점을 포함한 문자열을 정수로 변환
        opacity_value = 1.0  # 기본 투명도 설정
        # hovertext에서 측정값만 표시
        hovertext_values = [
            f"측정값 : {actual_values[i]:.1f}%"
            for i in range(len(batter_percentile_columns))
        ]
    else:
        percentile_values = similar_batters.iloc[0][batter_percentile_columns].astype(float).values  # 문자열을 실수형으로 변환
        text_values = ''  # 텍스트 출력 안 함
        opacity_value = 0.3  # 투명도를 연하게 설정
        # hovertext에서 측정값만 표시
        hovertext_values = [
            f"측정값 : {actual_values[i]:.1f}%"
            for i in range(len(batter_percentile_columns))
        ]

    # y축 레이블 설정
    y_labels = [label.replace('_std_percentile', '').replace('_', ' ').title() for label in batter_percentile_columns]

    # 막대그래프 생성
    fig = go.Figure([go.Bar(
        x=percentile_values,
        y=y_labels,
        orientation='h',
        text=text_values,  # 텍스트로 백분위수 값 출력 또는 생략
        textposition='outside',
        marker=dict(
            color=percentile_values,
            colorscale=[[0, 'blue'], [0.5, 'white'], [1, 'red']],
            cmin=0,
            cmax=100,
            line=dict(color='black', width=1),
            opacity=opacity_value  # 투명도 적용
        ),
        # 수정된 hovertext를 사용
        hovertext=hovertext_values,
        hoverinfo='text'
    )])

    # 기준선 추가
    for x in [5, 50, 95]:
        fig.add_shape(type="line",
                      x0=x, y0=-0.5, x1=x, y1=len(y_labels)-0.5,
                      line=dict(color="gray", width=1, dash="dash"))

    # 레이아웃 설정
    fig.update_layout(
        title="Plate Discipline - Percentile Ranking",
        title_x=0,
        title_xanchor='left',
        title_font=dict(size=20, family='Arial', color='black'),
        bargap=0.2,
        margin=dict(l=0, r=0, t=30, b=75),
        height=300,
        template="plotly_white",
        xaxis=dict(range=[0, 105]),
        yaxis=dict(
            automargin=True,
            showgrid=False,
            autorange='reversed'
        ))

    return fig

def plot_percentile_bar_chart_for_pitchers(similar_pitchers, pitcher_name_id_input):
    # 6개의 투수 Plate Discipline 백분위수 컬럼 리스트
    pitcher_percentile_columns = [
        'contact%_std_percentile', 'zone_contact%_std_percentile',
        'out_zone_swing%_std_percentile', 'first_pitch_strike%_std_percentile',
        'swstr%_std_percentile', 'csw%_std_percentile']

    pitcher_percentile_columns_adj = [
        'contact%_std_adj_percentile', 'zone_contact%_std_adj_percentile',
        'out_zone_swing%_std_adj_percentile', 'first_pitch_strike%_std_adj_percentile',
        'swstr%_std_adj_percentile', 'csw%_std_adj_percentile']

    # target_adj를 사용하여 pitcher_name_id_input의 정보를 가져옴
    target_adj = result_df_p[result_df_p['pitcher_name_id'] == pitcher_name_id_input].iloc[0]

    # 선수의 PA와 game_year를 가져옴
    target_pa = similar_pitchers.iloc[0]['pa']
    game_year = similar_pitchers.iloc[0]['game_year']

    # game_year가 문자열인 경우 정수로 변환
    if isinstance(game_year, str):
        game_year = int(game_year)

    # game_year에 따라 pa_threshold 설정
    if game_year == 2020:
        pa_threshold = 75
    else:
        pa_threshold = 202

    # 선수의 실제 측정값 추출
    actual_columns = ['contact%', 'zone_contact%', 'out_zone_swing%', 'first_pitch_strike%', 'swstr%', 'csw%']
    actual_values = similar_pitchers.iloc[0][actual_columns].values

    # PA 기준에 따라 백분위수 컬럼 선택
    if target_pa >= pa_threshold:
        percentile_values = target_adj[pitcher_percentile_columns_adj].astype(float).values  # 문자열을 실수형으로 변환
        text_values = [f"{int(float(value))}" for value in percentile_values]  # 소수점을 포함한 문자열을 정수로 변환
        opacity_value = 1.0  # 기본 투명도 설정
        # hovertext에서 측정값만 표시
        hovertext_values = [
            f"측정값 : {actual_values[i]:.1f}%"
            for i in range(len(pitcher_percentile_columns))
        ]
    else:
        percentile_values = similar_pitchers.iloc[0][pitcher_percentile_columns].astype(float).values  # 문자열을 실수형으로 변환
        text_values = ''  # 텍스트 출력 안 함
        opacity_value = 0.3  # 투명도를 연하게 설정
        # hovertext에서 측정값만 표시
        hovertext_values = [
            f"측정값 : {actual_values[i]:.1f}%"
            for i in range(len(pitcher_percentile_columns))
        ]

    # y축 레이블 설정
    y_labels = [label.replace('_std_percentile', '').replace('_', ' ').title() for label in pitcher_percentile_columns]

    # 막대그래프 생성 (백분위수 값을 텍스트로 막대에 표시)
    fig = go.Figure([go.Bar(
        x=percentile_values,
        y=y_labels,
        orientation='h',
        text=text_values,  # 텍스트로 "NOT QUALIFIED" 또는 백분위수 값 출력
        textposition='outside',  # 텍스트를 막대 바깥쪽에 표시
        marker=dict(
            color=percentile_values,  # 여전히 colorscale에 따라 색상이 적용됨
            colorscale=[[0, 'blue'], [0.5, 'white'], [1, 'red']],  # 낮으면 파란색, 중간 흰색, 높으면 빨간색
            cmin=0,
            cmax=100,
            line=dict(color='black', width=1),
            opacity=opacity_value  # 투명도를 적용
        ),
        # 수정된 hovertext를 사용
        hovertext=hovertext_values,
        hoverinfo='text'  # hovertext로 지정된 내용을 표시
    )])

    # 5%, 50%, 95% 지점에 선을 추가
    for x in [5, 50, 95]:
        fig.add_shape(type="line",
                      x0=x, y0=-0.5, x1=x, y1=len(y_labels)-0.5,
                      line=dict(color="gray", width=1, dash="dash"))

    # 그래프의 제목 및 레이아웃 설정
    fig.update_layout(
        title="Plate Discipline - Percentile Ranking",
        title_x=0,
        title_xanchor='left',
        title_font=dict(size=20, family='Arial', color='black'),
        bargap=0.2,
        margin=dict(l=0, r=0, t=30, b=75),  # 좌우 여백 설정
        height=300,  # 그래프의 높이 설정
        template="plotly_white",
        xaxis=dict(range=[0, 105]),  # x축 범위를 0~105로 설정
        yaxis=dict(
            automargin=True,  # 레이블 간격 조정
            showgrid=False,  # y축에 격자선 제거
            autorange='reversed'
        ))

    return fig

#-----------------------------------------------------------------------------------------------------------------------------

def plot_percentile_bar_chart_for_batters_batted_ball(similar_batters_bb, batter_name_id_input):
    # 2개의 타자 Batted Ball 백분위수 컬럼 리스트
    batter_percentile_columns = [
        'exitspeed_avg_std_percentile', 'exitspeed_max_std_percentile']

    batter_percentile_columns_adj = [
        'exitspeed_avg_std_adj_percentile', 'exitspeed_max_std_adj_percentile']

    # target_adj를 사용하여 batter_name_id_input의 정보를 가져옴
    target_adj = result_df[result_df['batter_name_id'] == batter_name_id_input].iloc[0]

    # 선수의 PA와 game_year를 가져옴
    target_pa = similar_batters_bb.iloc[0]['pa']
    game_year = similar_batters_bb.iloc[0]['game_year']

    # game_year가 문자열인 경우 정수로 변환
    if isinstance(game_year, str):
        game_year = int(game_year)

    # game_year에 따라 pa_threshold 설정
    if game_year == 2020:
        pa_threshold = 126
    else:
        pa_threshold = 340

    # 선수의 실제 측정값 추출
    actual_columns = ['exitspeed_avg', 'exitspeed_max']
    actual_values = similar_batters_bb.iloc[0][actual_columns].values

    # 선수의 PA가 기준 미만이면 투명도를 조정하고 텍스트 출력 안 함
    if target_pa >= pa_threshold:
        percentile_values = target_adj[batter_percentile_columns_adj].astype(float).values  # 문자열을 실수형으로 변환
        text_values = [f"{int(float(value))}" for value in percentile_values]  # 소수점을 포함한 문자열을 정수로 변환
        opacity_value = 1.0  # 기본 투명도 설정
        # hovertext에 측정값만 표시
        hovertext_values = [
            f"측정값 : {actual_values[i]:.1f}km/h"
            for i in range(len(batter_percentile_columns))
        ]
    else:
        percentile_values = similar_batters_bb.iloc[0][batter_percentile_columns].astype(float).values  # 문자열을 실수형으로 변환
        text_values = ''  # 텍스트 출력 안 함
        opacity_value = 0.3  # 투명도를 연하게 설정
        # hovertext에서 측정값만 표시
        hovertext_values = [
            f"측정값 : {actual_values[i]:.1f}km/h"
            for i in range(len(batter_percentile_columns))
        ]

    # y축 레이블 설정
    y_labels = [label.replace('_std_percentile', '').replace('_', ' ').title() for label in batter_percentile_columns]

    # 막대그래프 생성
    fig = go.Figure([go.Bar(
        x=percentile_values,
        y=y_labels,
        orientation='h',
        text=text_values,  # 텍스트로 백분위수 값 출력 또는 생략
        textposition='outside',
        marker=dict(
            color=percentile_values,
            colorscale=[[0, 'blue'], [0.5, 'white'], [1, 'red']],
            cmin=0,
            cmax=100,
            line=dict(color='black', width=1),
            opacity=opacity_value  # 투명도 적용
        ),
        # 수정된 hovertext를 사용
        hovertext=hovertext_values,
        hoverinfo='text'
    )])

    # 기준선 추가
    for x in [5, 50, 95]:
        fig.add_shape(type="line",
                      x0=x, y0=-0.5, x1=x, y1=len(y_labels)-0.5,
                      line=dict(color="gray", width=1, dash="dash"))

    # 레이아웃 설정
    fig.update_layout(
        title="Batted Ball - Percentile Ranking",
        title_x=0,
        title_xanchor='left',
        title_font=dict(size=20, family='Arial', color='black'),
        bargap=0.2,
        margin=dict(l=110, r=0, t=30, b=75),
        height=180,
        template="plotly_white",
        xaxis=dict(range=[0, 105]),
        yaxis=dict(
            automargin=True,
            showgrid=False,
            autorange='reversed'
        ))

    return fig

def plot_percentile_bar_chart_for_pitchers_batted_ball(similar_pitchers_bb, pitcher_name_id_input):
    # 2개의 투수 Batted Ball 백분위수 컬럼 리스트
    pitcher_percentile_columns = [
        'exitspeed_avg_std_percentile', 'exitspeed_max_std_percentile']

    pitcher_percentile_columns_adj = [
        'exitspeed_avg_std_adj_percentile', 'exitspeed_max_std_adj_percentile']

    # target_adj를 사용하여 pitcher_name_id_input의 정보를 가져옴
    target_adj = result_df_p[result_df_p['pitcher_name_id'] == pitcher_name_id_input].iloc[0]

    # 선수의 PA와 game_year 가져옴
    target_pa = similar_pitchers_bb.iloc[0]['pa']
    game_year = similar_pitchers_bb.iloc[0]['game_year']

    # game_year가 문자열인 경우 정수로 변환
    if isinstance(game_year, str):
        game_year = int(game_year)

    # game_year에 따라 pa_threshold 설정
    if game_year == 2020:
        pa_threshold = 75
    else:
        pa_threshold = 202

    # 선수의 실제 측정값 추출
    actual_columns = ['exitspeed_avg', 'exitspeed_max']
    actual_values = similar_pitchers_bb.iloc[0][actual_columns].values

    # 선수의 PA가 기준 미만이면 투명도를 조정하고 텍스트 출력 안 함
    if target_pa >= pa_threshold:
        percentile_values = target_adj[pitcher_percentile_columns_adj].astype(float).values  # 문자열을 실수형으로 변환
        text_values = [f"{int(float(value))}" for value in percentile_values]  # 소수점을 포함한 문자열을 정수로 변환
        opacity_value = 1.0  # 기본 투명도 설정
        # hovertext에 측정값만 표시
        hovertext_values = [
            f"측정값 : {actual_values[i]:.1f}km/h"
            for i in range(len(pitcher_percentile_columns))
        ]
    else:
        percentile_values = similar_pitchers_bb.iloc[0][pitcher_percentile_columns].astype(float).values  # 문자열을 실수형으로 변환
        text_values = ''
        opacity_value = 0.3  # 투명도를 연하게 설정
        # hovertext에서 백분위수는 제외하고 측정값만 표시
        hovertext_values = [
            f"측정값 : {actual_values[i]:.1f}km/h"
            for i in range(len(pitcher_percentile_columns))
        ]

    # y축 레이블 설정
    y_labels = [label.replace('_std_percentile', '').replace('_', ' ').title() for label in pitcher_percentile_columns]

    # 막대그래프 생성 (백분위수 값을 텍스트로 막대에 표시)
    fig = go.Figure([go.Bar(
        x=percentile_values,
        y=y_labels,
        orientation='h',
        text=text_values,  # 텍스트로 "NOT QUALIFIED" 또는 백분위수 값 출력
        textposition='outside',  # 텍스트를 막대 바깥쪽에 표시
        marker=dict(
            color=percentile_values,  # 여전히 colorscale에 따라 색상이 적용됨
            colorscale=[[0, 'blue'], [0.5, 'white'], [1, 'red']],  # 낮으면 파란색, 중간 흰색, 높으면 빨간색
            cmin=0,
            cmax=100,
            line=dict(color='black', width=1),
            opacity=opacity_value  # 투명도를 적용
        ),
        # 수정된 hovertext를 사용
        hovertext=hovertext_values,
        hoverinfo='text'  # hovertext로 지정된 내용을 표시
    )])

    # 5%, 50%, 95% 지점에 선을 추가
    for x in [5, 50, 95]:
        fig.add_shape(type="line",
                      x0=x, y0=-0.5, x1=x, y1=len(y_labels)-0.5,
                      line=dict(color="gray", width=1, dash="dash"))

    # 그래프의 제목 및 레이아웃 설정
    fig.update_layout(
        title="Plate Discipline - Percentile Ranking",
        title_x=0,
        title_xanchor='left',
        title_font=dict(size=20, family='Arial', color='black'),
        bargap=0.2,
        margin=dict(l=110, r=0, t=30, b=75),  # 좌우 여백 설정
        height=180,  # 그래프의 높이 설정
        template="plotly_white",
        xaxis=dict(range=[0, 105]),  # x축 범위를 0~105로 설정
        yaxis=dict(
            automargin=True,  # 레이블 간격 조정
            showgrid=False,  # y축에 격자선 제거
            autorange='reversed'
        ))

    return fig

#-----------------------------------------------------------------------------------------------------------------------------

# 타자 Plate Discipline 스탯 평균과의 차이
def plot_difference_from_avg_for_batters(similar_batters, df_swing_table_all, batter_name, game_year):
    # 해당 game_year에 맞는 평균값을 df_swing_table_all에서 추출
    df_year = df_swing_table_all[df_swing_table_all['game_year'] == game_year].iloc[0]

    # 입력된 선수의 차이 값과 실제 측정값 추출 (첫 번째 행을 기준으로, 역순으로 정렬)
    player_diffs = similar_batters.iloc[0][[
        'out_zone_swing%_diff_from_avg', 'zone_swing%_diff_from_avg', 'swing%_diff_from_avg',
        'zone%_diff_from_avg', 'out_zone_contact%_diff_from_avg', 'zone_contact%_diff_from_avg',
        'contact%_diff_from_avg', 'first_pitch_swing%_diff_from_avg', 'swstr%_diff_from_avg', 'cstr%_diff_from_avg', 'csw%_diff_from_avg'
    ]][::-1].astype(float)  # 데이터 타입을 float로 변환하여 역순으로 정렬

    # 선수의 실제 측정값 추출 (역순으로 정렬)
    player_values = similar_batters.iloc[0][[
        'out_zone_swing%','zone_swing%','swing%','zone%','out_zone_contact%','zone_contact%',
        'contact%','first_pitch_swing%','swstr%','cstr%','csw%'
    ]][::-1].astype(float)  # 데이터 타입을 float로 변환하여 역순으로 정렬

    # 각 지표의 해당 연도 평균값을 추출 (역순으로 정렬)
    mean_values = df_year[[
        'out_zone_swing%','zone_swing%','swing%','zone%','out_zone_contact%','zone_contact%',
        'contact%','first_pitch_swing%','swstr%','cstr%','csw%'
    ]][::-1].astype(float)

    # y축에서 '_diff_from_avg' 제거하고 첫 글자를 대문자로 바꾸기
    y_labels = [label.replace('_diff_from_avg', '').replace('_', ' ').title() for label in player_diffs.index]

    # 각 막대에 마우스를 올렸을 때 나타날 텍스트 설정 (지표 이름과 실제 측정값)
    hover_texts = [
        f"{y_label} : {player_values[i]:.1f}%<br>리그 평균 : {mean_values[i]:.1f}%"
        for i, y_label in enumerate(y_labels)]

    # 막대그래프 생성 (레이블에는 차이를 표시, hovertext에는 실제 컬럼 이름과 값을 표시)
    fig = go.Figure([go.Bar(
        x=player_diffs.values,
        y=y_labels,  # y축 값에 포맷된 값 적용
        orientation='h',
        text=[f"{diff:.1f}" for diff in player_diffs.values],  # 차이를 텍스트로 표시
        textposition='outside',  # 텍스트 위치를 막대 바깥으로 설정
        hovertext=hover_texts,  # 마우스 hover 시 실제 컬럼 이름과 값 표시
        hoverinfo='text',  # hover에서 실제 컬럼 이름과 값만 표시
        marker=dict(
            color=player_diffs.values,  # 막대의 색상을 차이에 따라 설정
            colorscale=custom_red_blue,  # 커스텀 빨강-파랑 색상 스케일 적용
            cmin=-30,
            cmax=30,
            line=dict(color='black', width=2),  # 테두리 색상과 두께 설정
            colorbar=dict(  # 컬러바 설정
                titleside="right",  # 제목 위치 설정
                ticks="outside"  # 컬러바 눈금 설정
            )
        )
    )])

    # 그래프의 제목과 축 레이블, 축 범위 설정
    fig.update_layout(
        title=f"Plate Discipline - {game_year}시즌 평균값과의 차이",  # 선수 이름과 연도 추가
        title_x=0,  # 제목의 x축 위치 설정 (0 = 왼쪽 정렬)
        title_xanchor='left',  # 제목을 왼쪽 정렬로 설정
        title_font=dict(size=20, family='Arial', color='black'),
        xaxis_range=[-30, 30],  # x축 범위를 -30에서 30으로 고정
        bargap=0.2,
        margin=dict(l=150, r=50, t=50, b=100),  # 그래프의 간격을 최소화
        height=600,  # 그래프의 높이 설정
        template="plotly_white")

    return fig

# 투수 Plate Discipline 스탯 평균과의 차이
def plot_difference_from_avg_for_pitchers(similar_pitchers, df_swing_table_all, pitcher_name, game_year):
    # 해당 game_year에 맞는 평균값을 df_swing_table_all에서 추출
    df_year = df_swing_table_all[df_swing_table_all['game_year'] == game_year].iloc[0]

    # 입력된 선수의 차이 값과 실제 측정값 추출 (첫 번째 행을 기준으로, 역순으로 정렬)
    player_diffs = similar_pitchers.iloc[0][[
        'out_zone_swing%_diff_from_avg', 'zone_swing%_diff_from_avg', 'swing%_diff_from_avg',
        'zone%_diff_from_avg', 'out_zone_contact%_diff_from_avg', 'zone_contact%_diff_from_avg',
        'contact%_diff_from_avg', 'first_pitch_strike%_diff_from_avg', 'swstr%_diff_from_avg', 'cstr%_diff_from_avg', 'csw%_diff_from_avg'
    ]][::-1].astype(float)  # 데이터 타입을 float로 변환하여 역순으로 정렬

    # 선수의 실제 측정값 추출 (역순으로 정렬)
    player_values = similar_pitchers.iloc[0][[
        'out_zone_swing%','zone_swing%','swing%','zone%','out_zone_contact%','zone_contact%',
        'contact%','first_pitch_strike%','swstr%','cstr%','csw%'
    ]][::-1].astype(float)  # 데이터 타입을 float로 변환하여 역순으로 정렬

    # 각 지표의 해당 연도 평균값을 추출 (역순으로 정렬)
    mean_values = df_year[[
        'out_zone_swing%','zone_swing%','swing%','zone%','out_zone_contact%','zone_contact%',
        'contact%','first_pitch_strike%','swstr%','cstr%','csw%'
    ]][::-1].astype(float)

    # y축에서 '_diff_from_avg' 제거하고 첫 글자를 대문자로 바꾸기
    y_labels = [label.replace('_diff_from_avg', '').replace('_', ' ').title() for label in player_diffs.index]

    # 각 막대에 마우스를 올렸을 때 나타날 텍스트 설정 (지표 이름과 실제 측정값)
    hover_texts = [
        f"{y_label} : {player_values[i]:.1f}%<br>리그 평균 : {mean_values[i]:.1f}%"
        for i, y_label in enumerate(y_labels)]

    # 막대그래프 생성 (레이블에는 차이를 표시, hovertext에는 실제 컬럼 이름과 값을 표시)
    fig = go.Figure([go.Bar(
        x=player_diffs.values,
        y=y_labels,  # y축 값에 포맷된 값 적용
        orientation='h',
        text=[f"{diff:.1f}" for diff in player_diffs.values],  # 차이를 텍스트로 표시
        textposition='outside',  # 텍스트 위치를 막대 바깥으로 설정
        hovertext=hover_texts,  # 마우스 hover 시 실제 컬럼 이름과 값 표시
        hoverinfo='text',  # hover에서 실제 컬럼 이름과 값만 표시
        marker=dict(
            color=player_diffs.values,  # 막대의 색상을 차이에 따라 설정
            colorscale=custom_red_blue,  # 커스텀 빨강-파랑 색상 스케일 적용
            cmin=-30,
            cmax=30,
            line=dict(color='black', width=2),  # 테두리 색상과 두께 설정
            colorbar=dict(  # 컬러바 설정
                titleside="right",  # 제목 위치 설정
                ticks="outside"  # 컬러바 눈금 설정
            )
        )
    )])

    # 그래프의 제목과 축 레이블, 축 범위 설정
    fig.update_layout(
        title=f"Plate Discipline - {game_year}시즌 평균값과의 차이",  # 선수 이름과 연도 추가
        title_x=0,  # 제목의 x축 위치 설정 (0 = 왼쪽 정렬)
        title_xanchor='left',  # 제목을 왼쪽 정렬로 설정
        title_font=dict(size=20, family='Arial', color='black'),
        xaxis_range=[-30, 30],  # x축 범위를 -30에서 30으로 고정
        bargap=0.2,
        margin=dict(l=150, r=50, t=50, b=100),  # 그래프의 간격을 최소화
        height=600,  # 그래프의 높이 설정
        template="plotly_white")

    return fig

#-----------------------------------------------------------------------------------------------------------------------------

# 타자 GB%, FB%, LD%, PU% 평균과의 차이
def plot_difference_from_avg_for_batters_batted_ball(similar_batters_bb, df_swing_table_all, batter_name, game_year):
    # 해당 game_year에 맞는 평균값을 df_swing_table_all에서 추출
    df_year = df_swing_table_all[df_swing_table_all['game_year'] == game_year].iloc[0]

    # 입력된 선수의 차이 값과 실제 측정값 추출 (첫 번째 행을 기준으로, 역순으로 정렬)
    player_diffs = similar_batters_bb.iloc[0][[
        'gb%_diff_from_avg', 'fb%_diff_from_avg', 'ld%_diff_from_avg', 'pu%_diff_from_avg'
    ]][::-1].astype(float)  # 데이터 타입을 float로 변환하여 역순으로 정렬

    # 선수의 실제 측정값 추출 (역순으로 정렬)
    player_values = similar_batters_bb.iloc[0][[
        'gb%','fb%','ld%','pu%'
    ]][::-1].astype(float)  # 데이터 타입을 float로 변환하여 역순으로 정렬

    # 각 지표의 해당 연도 평균값을 추출 (역순으로 정렬)
    mean_values = df_year[[
        'gb%','fb%','ld%','pu%'
    ]][::-1].astype(float)

    # y축에서 '_diff_from_avg' 제거하고 첫 글자를 대문자로 바꾸기
    y_labels = [label.replace('_diff_from_avg', '').replace('_', ' ').upper() for label in player_diffs.index]

    # 각 막대에 마우스를 올렸을 때 나타날 텍스트 설정 (지표 이름과 실제 측정값)
    hover_texts = [
        f"{y_label} : {player_values[i]:.1f}%<br>리그 평균 : {mean_values[i]:.1f}%"
        for i, y_label in enumerate(y_labels)]

    # 막대그래프 생성 (레이블에는 차이를 표시, hovertext에는 실제 컬럼 이름과 값을 표시)
    fig = go.Figure([go.Bar(
        x=player_diffs.values,
        y=y_labels,  # y축 값에 포맷된 값 적용
        orientation='h',
        text=[f"{diff:.1f}" for diff in player_diffs.values],  # 차이를 텍스트로 표시
        textposition='outside',  # 텍스트 위치를 막대 바깥으로 설정
        hovertext=hover_texts,  # 마우스 hover 시 실제 컬럼 이름과 값 표시
        hoverinfo='text',  # hover에서 실제 컬럼 이름과 값만 표시
        marker=dict(
            color=player_diffs.values,  # 막대의 색상을 차이에 따라 설정
            colorscale=custom_red_blue,  # 커스텀 빨강-파랑 색상 스케일 적용
            cmin=-30,
            cmax=30,
            line=dict(color='black', width=2),  # 테두리 색상과 두께 설정
            colorbar=dict(  # 컬러바 설정
                titleside="right",  # 제목 위치 설정
                ticks="outside"  # 컬러바 눈금 설정
            )
        )
    )])

    # 그래프의 제목과 축 레이블, 축 범위 설정
    fig.update_layout(
        title=f"Batted Ball - {game_year}시즌 평균값과의 차이",  # 선수 이름과 연도 추가
        title_x=0,  # 제목의 x축 위치 설정 (0 = 왼쪽 정렬)
        title_xanchor='left',  # 제목을 왼쪽 정렬로 설정
        title_font=dict(size=20, family='Arial', color='black'),
        xaxis_range=[-20, 20],  # x축 범위를 -20에서 20으로 고정
        bargap=0.2,
        margin=dict(l=150, r=50, t=50, b=100),  # 그래프의 간격을 최소화
        height=300,  # 그래프의 높이 설정
        template="plotly_white")

    return fig

# 투수 GB%, FB%, LD%, PU% 평균과의 차이
def plot_difference_from_avg_for_pitchers_batted_ball(similar_pitchers_bb, df_swing_table_all, pitcher_name, game_year):
    # 해당 game_year에 맞는 평균값을 df_swing_table_all에서 추출
    df_year = df_swing_table_all[df_swing_table_all['game_year'] == game_year].iloc[0]

    # 입력된 선수의 차이 값과 실제 측정값 추출 (첫 번째 행을 기준으로, 역순으로 정렬)
    player_diffs = similar_pitchers_bb.iloc[0][[
        'gb%_diff_from_avg', 'fb%_diff_from_avg', 'ld%_diff_from_avg', 'pu%_diff_from_avg'
    ]][::-1].astype(float)  # 데이터 타입을 float로 변환하여 역순으로 정렬

    # 선수의 실제 측정값 추출 (역순으로 정렬)
    player_values = similar_pitchers_bb.iloc[0][[
        'gb%','fb%','ld%','pu%'
    ]][::-1].astype(float)  # 데이터 타입을 float로 변환하여 역순으로 정렬

    # 각 지표의 해당 연도 평균값을 추출 (역순으로 정렬)
    mean_values = df_year[[
        'gb%','fb%','ld%','pu%'
    ]][::-1].astype(float)

    # y축에서 '_diff_from_avg' 제거하고 첫 글자를 대문자로 바꾸기
    y_labels = [label.replace('_diff_from_avg', '').replace('_', ' ').upper() for label in player_diffs.index]

    # 각 막대에 마우스를 올렸을 때 나타날 텍스트 설정 (지표 이름과 실제 측정값)
    hover_texts = [
        f"{y_label} : {player_values[i]:.1f}%<br>리그 평균 : {mean_values[i]:.1f}%"
        for i, y_label in enumerate(y_labels)]

    # 막대그래프 생성 (레이블에는 차이를 표시, hovertext에는 실제 컬럼 이름과 값을 표시)
    fig = go.Figure([go.Bar(
        x=player_diffs.values,
        y=y_labels,  # y축 값에 포맷된 값 적용
        orientation='h',
        text=[f"{diff:.1f}" for diff in player_diffs.values],  # 차이를 텍스트로 표시
        textposition='outside',  # 텍스트 위치를 막대 바깥으로 설정
        hovertext=hover_texts,  # 마우스 hover 시 실제 컬럼 이름과 값 표시
        hoverinfo='text',  # hover에서 실제 컬럼 이름과 값만 표시
        marker=dict(
            color=player_diffs.values,  # 막대의 색상을 차이에 따라 설정
            colorscale=custom_red_blue,  # 커스텀 빨강-파랑 색상 스케일 적용
            cmin=-30,
            cmax=30,
            line=dict(color='black', width=2),  # 테두리 색상과 두께 설정
            colorbar=dict(  # 컬러바 설정
                titleside="right",  # 제목 위치 설정
                ticks="outside"  # 컬러바 눈금 설정
            )
        )
    )])

    # 그래프의 제목과 축 레이블, 축 범위 설정
    fig.update_layout(
        title=f"Batted Ball - {game_year}시즌 평균값과의 차이",  # 선수 이름과 연도 추가
        title_x=0,  # 제목의 x축 위치 설정 (0 = 왼쪽 정렬)
        title_xanchor='left',  # 제목을 왼쪽 정렬로 설정
        title_font=dict(size=20, family='Arial', color='black'),
        xaxis_range=[-15, 15],  # x축 범위를 -15에서 15로 고정
        bargap=0.2,
        margin=dict(l=150, r=50, t=50, b=100),  # 그래프의 간격을 최소화
        height=300,  # 그래프의 높이 설정
        template="plotly_white")

    return fig

#-----------------------------------------------------------------------------------------------------------------------------

def plot_combined_euclidean_distance_scatter_batter(similar_batters, similar_batters_bb):
    # 두 데이터프레임을 여러 열을 기준으로 병합
    merge_columns = ['batter_name_id', 'batter', 'batter_name_new', 'game_year', 'pa']
    merged_df = similar_batters.merge(similar_batters_bb, on=merge_columns, suffixes=('_batters', '_batters_bb'))

    # euclidean_distance_batters나 euclidean_distance_batters_bb가 0인 행 찾기
    zero_distance_rows = merged_df[(merged_df['euclidean_distance_batters'] == 0) | (merged_df['euclidean_distance_batters_bb'] == 0)]

    # 각 행과 0 거리 행 사이의 유클리드 거리 계산
    merged_df['distance_to_zero'] = merged_df.apply(
        lambda row: np.min([
            np.sqrt((row['euclidean_distance_batters'] - zero_row['euclidean_distance_batters']) ** 2 +
                    (row['euclidean_distance_batters_bb'] - zero_row['euclidean_distance_batters_bb']) ** 2)
            for _, zero_row in zero_distance_rows.iterrows()
        ]) if not zero_distance_rows.empty else np.nan,
        axis=1
    )

    # Distance to Zero가 0이 아닌 가장 낮은 5명과 가장 큰 5명 선택
    top_5_closest = merged_df[merged_df['distance_to_zero'] > 0].nsmallest(5, 'distance_to_zero')
    top_5_farthest = merged_df.nlargest(5, 'distance_to_zero')

    # 산점도 생성 - 기본 마커
    fig = go.Figure(data=go.Scatter(
        x=merged_df['euclidean_distance_batters'],
        y=merged_df['euclidean_distance_batters_bb'],
        mode='markers',
        marker=dict(
            size=8,
            color=merged_df['distance_to_zero'],  # 0 거리 행까지의 거리에 따라 색상 지정
            colorscale='Agsunset',
            showscale=True  # 색상 범례 표시
        ),
        hovertext=[
            f"{row['batter_name_new']} ({int(row['game_year'])})<br>"
            f"PA : {row['pa']}<br>"
            if row['distance_to_zero'] == 0 else
            f"{row['batter_name_new']} ({int(row['game_year'])})<br>"
            f"PA : {row['pa']}<br>"
            f"Plate Discipline Euclidean Distance : {row['euclidean_distance_batters']:.3f}<br>"
            f"Batted Ball Euclidean Distance : {row['euclidean_distance_batters_bb']:.3f}<br>"
            f"Similarity : {row['distance_to_zero']:.3f}"
            for _, row in merged_df.iterrows()
        ],
        hoverinfo='text',
        showlegend=False  # 범례 표시 안 함
    ))

    # Distance to Zero가 가장 낮은 5명에게 화살표 추가 (텍스트 없이)
    annotations = []
    for _, row in top_5_closest.iterrows():
        annotations.append(dict(
            x=row['euclidean_distance_batters'],
            y=row['euclidean_distance_batters_bb'],
            ax=0,  # 화살표 시작 x 위치 (0,0)에서 시작하도록 설정
            ay=0,  # 화살표 시작 y 위치
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor="black"
        ))

    # 레이아웃 업데이트 및 화살표 추가
    fig.update_layout(
        title="Comparison of Similar Batters (Plate Discipline & Batted Ball)",
        title_x=0,
        title_xanchor='left',
        title_font=dict(size=20, family='Arial', color='black'),
        xaxis=dict(title='Plate Discipline Euclidean Distance', showticklabels=False),
        yaxis=dict(title='Batted Ball Euclidean Distance', showticklabels=False),
        height=650,
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=40),
        annotations=annotations  # 화살표 추가
    )

    return fig

def plot_combined_euclidean_distance_scatter_pitcher(similar_pitchers, similar_pitchers_bb):
    # 두 데이터프레임을 여러 열을 기준으로 병합
    merge_columns = ['pitcher_name_id', 'pitcher', 'pitcher_name_new', 'game_year', 'pa']
    merged_df = similar_pitchers.merge(similar_pitchers_bb, on=merge_columns, suffixes=('_pitchers', '_pitchers_bb'))

    # euclidean_distance_pitchers나 euclidean_distance_pitchers_bb가 0인 행 찾기
    zero_distance_rows = merged_df[(merged_df['euclidean_distance_pitchers'] == 0) | (merged_df['euclidean_distance_pitchers_bb'] == 0)]

    # 각 행과 0 거리 행 사이의 유클리드 거리 계산
    merged_df['distance_to_zero'] = merged_df.apply(
        lambda row: np.min([
            np.sqrt((row['euclidean_distance_pitchers'] - zero_row['euclidean_distance_pitchers']) ** 2 +
                    (row['euclidean_distance_pitchers_bb'] - zero_row['euclidean_distance_pitchers_bb']) ** 2)
            for _, zero_row in zero_distance_rows.iterrows()
        ]) if not zero_distance_rows.empty else np.nan,
        axis=1
    )

    # Distance to Zero가 0이 아닌 가장 낮은 5명과 가장 큰 5명 선택
    top_5_closest = merged_df[merged_df['distance_to_zero'] > 0].nsmallest(5, 'distance_to_zero')
    top_5_farthest = merged_df.nlargest(5, 'distance_to_zero')

    # 산점도 생성 - 기본 마커
    fig = go.Figure(data=go.Scatter(
        x=merged_df['euclidean_distance_pitchers'],
        y=merged_df['euclidean_distance_pitchers_bb'],
        mode='markers',
        marker=dict(
            size=8,
            color=merged_df['distance_to_zero'],  # 0 거리 행까지의 거리에 따라 색상 지정
            colorscale='Agsunset',
            showscale=True  # 색상 범례 표시
        ),
        hovertext=[
            f"{row['pitcher_name_new']} ({int(row['game_year'])})<br>"
            f"PA : {row['pa']}<br>"
            if row['distance_to_zero'] == 0 else
            f"{row['pitcher_name_new']} ({int(row['game_year'])})<br>"
            f"PA : {row['pa']}<br>"
            f"Plate Discipline Euclidean Distance : {row['euclidean_distance_pitchers']:.3f}<br>"
            f"Batted Ball Euclidean Distance : {row['euclidean_distance_pitchers_bb']:.3f}<br>"
            f"Similarity : {row['distance_to_zero']:.3f}"
            for _, row in merged_df.iterrows()
        ],
        hoverinfo='text',
        showlegend=False  # 범례 표시 안 함
    ))

    # Distance to Zero가 가장 낮은 5명에게 화살표 추가 (텍스트 없이)
    annotations = []
    for _, row in top_5_closest.iterrows():
        annotations.append(dict(
            x=row['euclidean_distance_pitchers'],
            y=row['euclidean_distance_pitchers_bb'],
            ax=0,  # 화살표 시작 x 위치 (0,0)에서 시작하도록 설정
            ay=0,  # 화살표 시작 y 위치
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor="black"
        ))

    # 레이아웃 업데이트 및 화살표 추가
    fig.update_layout(
        title="Comparison of Similar Pitchers (Plate Discipline & Batted Ball)",
        title_x=0,
        title_xanchor='left',
        title_font=dict(size=20, family='Arial', color='black'),
        xaxis=dict(title='Plate Discipline Euclidean Distance', showticklabels=False),
        yaxis=dict(title='Batted Ball Euclidean Distance', showticklabels=False),
        height=650,
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=40),
        annotations=annotations  # 화살표 추가
    )

    return fig

# 애플리케이션 실행
if __name__ == '__main__':
    app.run_server(debug=True)
#-----------------------------------------------------------------------------------------------------------------------------