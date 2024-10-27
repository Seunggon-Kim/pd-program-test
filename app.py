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
from dash.dash_table import DataTable
import pandas as pd
import plotly.graph_objects as go
import scipy
from scipy.spatial.distance import euclidean

# Dash 애플리케이션 생성
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = ('Dashboard | Plate Discipline')
server = app.server

# 설명 레이어 (레이어 0)
from dash import dcc, html
from dash.dash_table import DataTable

layer0_layout = html.Div([
    html.H1("Plate Discipline 유사도 프로그램", style={'margin-bottom': '20px'}),
    html.P("안녕하세요, Plate Discipline 유사도 프로그램을 방문해주셔서 감사합니다.", style={'margin-bottom': '20px'}),
    
    html.P("해당 프로그램은 타자와 투수의 이름을 입력하면 유클리드 거리에 기반해서 입력한 선수와 유사한 선수들을 확인하실 수 있습니다.", style={'margin-bottom': '30px'}),
    
    html.H2("Plate Discipline이란?", style={'margin-top': '40px', 'margin-bottom': '10px'}),
    html.P("타자가 타석에서 스트라이크존을 빠르게 파악하고, 침착하게 대응하는 능력을 뜻합니다. "
           "플레이트 디서플린은 타석에서 적응력과 참을성, 좋지 않은 공을 거를 수 있는 능력입니다. "
           "우열을 가리기보다는 타자의 캐릭터에 가까운 지표라고 볼 수 있습니다.", style={'margin-bottom': '30px'}),
    
    html.H2("유클리드 거리란?", style={'margin-top': '40px', 'margin-bottom': '10px'}),
    html.P("유클리드 거리는 두 점 사이의 직선 거리를 계산하는 방법으로, 두 점 간의 좌표 차이를 제곱하여 합산하고, 그 합의 제곱근을 구하는 방식으로 계산됩니다. "
           "이 프로그램에서는 선수의 Plate Discipline 지표들을 각각의 좌표로 간주하여 유클리드 거리를 계산하며, 이를 통해 두 선수의 유사성을 측정합니다. "
           "유클리드 거리가 짧을수록 두 선수의 Plate Discipline 특성이 유사하다고 해석할 수 있습니다.", style={'margin-bottom': '30px'}),
    
    html.H2("유사도 계산에 사용된 지표, 계산 방법", style={'margin-top': '40px', 'margin-bottom': '10px'}),
    
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
    html.Div(id='batter-percentile-bar-chart-container'),
    html.Div(id='batter-difference-from-avg-chart-container'),
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
    html.Div(id='pitcher-percentile-bar-chart-container'),
    html.Div(id='pitcher-difference-from-avg-chart-container'),
    html.Div(id='pitcher-euclidean-distance-scatter-container'),  # Euclidean distance scatter container 추가
])

# 탭을 사용하여 세 개의 레이어를 구분
app.layout = html.Div([
    dcc.Tabs(id='tabs', value='layer0', children=[
        dcc.Tab(label='시작하기에 앞서', value='layer0'),
        dcc.Tab(label='타자 Plate Discipline 유사도 분석', value='layer1'),
        dcc.Tab(label='투수 Plate Discipline 유사도 분석', value='layer2'),
    ]),
    html.Div(id='tabs-content')
])

#-----------------------------------------------------------------------------------------------------------------------------

# 타자 유사도를 계산할 표준 정규화된 컬럼 리스트
columns_to_compare_batter = [
    'out_zone_swing%_std', 'zone_swing%_std', 'swing%_std',
    'out_zone_contact%_std', 'zone_contact%_std',
    'contact%_std', 'first_pitch_swing%_std',
    'swstr%_std', 'cstr%_std', 'csw%_std'
]

# 함께 출력할 원래 컬럼 리스트
original_columns_batter = [
    'out_zone_swing%', 'zone_swing%', 'swing%', 'zone%',
    'out_zone_contact%', 'zone_contact%', 'contact%',
    'first_pitch_strike%', 'first_pitch_swing%', 'swstr%',
    'cstr%', 'csw%'
]

# 추가할 백분위수 컬럼 리스트
percentile_columns_batter = [f"{col}_percentile" for col in columns_to_compare_batter]

# 투수 유사도를 계산할 표준 정규화된 컬럼 리스트
columns_to_compare_pitcher = [
    'out_zone_swing%_std', 'zone_swing%_std', 'swing%_std',
    'out_zone_contact%_std', 'zone_contact%_std',
    'contact%_std', 'first_pitch_strike%_std',
    'swstr%_std', 'cstr%_std', 'csw%_std'
]

# 함께 출력할 원래 컬럼 리스트
original_columns_pitcher = [
    'out_zone_swing%', 'zone_swing%', 'swing%', 'zone%',
    'out_zone_contact%', 'zone_contact%', 'contact%',
    'first_pitch_strike%', 'first_pitch_swing%', 'swstr%', 'cstr%', 'csw%'
]

# 추가할 백분위수 컬럼 리스트
percentile_columns_pitcher = [f"{col}_percentile" for col in columns_to_compare_pitcher]

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

# 타자 유사도 계산 함수
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
    elif 100 < target_pa <= 200:
        result_df_pa = df_exclude_self[(df_exclude_self['pa'] >= target_pa - 50) & (df_exclude_self['pa'] <= target_pa + 50)]
    elif target_pa > 200:
        result_df_pa = df_exclude_self[(df_exclude_self['pa'] >= target_pa - 100) & (df_exclude_self['pa'] <= target_pa + 100)]
    else:
        print("Invalid pa range.")
        return None

    result_with_target = pd.concat([target_row_with_distance.to_frame().T, result_df_pa], ignore_index=True).reset_index(drop=True)

    return result_with_target[['batter_name_id', 'batter', 'batter_name_new', 'game_year', 'pa', 'euclidean_distance', 'similarity_percent']
                              + original_columns_batter + percentile_columns_batter + [f"{col}_diff_from_avg" for col in original_columns_batter]]

    similar_batters = calculate_batter_similarity(batter_name_id_input)

    # 결과 확인
    similar_batters_top5 = similar_batters.sort_values(by='euclidean_distance', ascending=True).head(6)

# 투수 유사도 계산 함수
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
    elif 100 < target_pa <= 200:
        result_df_pa = df_exclude_self[(df_exclude_self['pa'] >= target_pa - 50) & (df_exclude_self['pa'] <= target_pa + 50)]
    elif target_pa > 200:
        result_df_pa = df_exclude_self[(df_exclude_self['pa'] >= target_pa - 100) & (df_exclude_self['pa'] <= target_pa + 100)]
    else:
        print("Invalid pa range.")
        return None

    result_with_target = pd.concat([target_row_with_distance.to_frame().T, result_df_pa], ignore_index=True).reset_index(drop=True)

    return result_with_target[['pitcher_name_id', 'pitcher', 'pitcher_name_new', 'game_year', 'pa', 'euclidean_distance', 'similarity_percent']
                              + original_columns_pitcher + percentile_columns_pitcher + [f"{col}_diff_from_avg" for col in original_columns_pitcher]]

    similar_pitchers = calculate_pitcher_similarity(pitcher_name_id_input)

    # 결과 확인
    similar_pitchers_top5 = similar_pitchers.sort_values(by='euclidean_distance', ascending=True).head(6)

# 커스텀 빨강-파랑 색상 스케일 정의
custom_red_blue = [
    [0, 'blue'],   # 최소값은 파란색
    [0.5, 'white'], # 중앙값은 흰색
    [1, 'red']     # 최대값은 빨간색
]

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

@app.callback(
    [Output('batter-output-container', 'children'),
     Output('batter-similarity-graph-container', 'children'),
     Output('batter-percentile-bar-chart-container', 'children'),
     Output('batter-difference-from-avg-chart-container', 'children'),
     Output('batter-euclidean-distance-scatter-container', 'children')],
    [Input('batter-dropdown', 'value')]
)
def update_batter_output(batter_name_id_input):
    if batter_name_id_input is None:
        return "선수, 시즌을 선택해주세요 (MLB ID 6자리로도 검색 가능합니다)", no_update, no_update, no_update, no_update

    # 유사도 분석 함수 호출
    similar_batters = calculate_batter_similarity(batter_name_id_input)

    if similar_batters is None or similar_batters.empty:
        return f"{batter_name_id_input} 선수의 데이터를 찾을 수 없습니다.", no_update, no_update, no_update, no_update

    # 결과를 필터링하여 상위 5명 추출
    similar_batters_top5 = similar_batters.sort_values(by='euclidean_distance').head(6)
    max_distance = similar_batters['euclidean_distance'].max()

    # 테이블 생성
    similarity_table_batter = dcc.Graph(figure=display_table_for_batters(similar_batters_top5))

    # 백분위수 그래프
    percentile_bar_chart_batter = dcc.Graph(figure=plot_percentile_bar_chart_for_batters(similar_batters_top5))

    # 평균과의 차이 그래프
    first_batter_row = similar_batters_top5.iloc[0]
    batter_name = first_batter_row['batter_name_new']
    game_year_batter = int(first_batter_row['game_year'])
    difference_from_avg_chart_batter = dcc.Graph(figure=plot_difference_from_avg_for_batters(similar_batters_top5, df_swing_table_all, batter_name, game_year_batter))

    # 유클리디안 거리 산포도
    euclidean_distance_scatter_batter = dcc.Graph(figure=plot_euclidean_distance_scatter_for_batters(similar_batters, [0, max_distance]))

    # 출력 메시지
    output_message_batter = f"{game_year_batter} 시즌 {batter_name}의 유사도 분석 결과입니다."

    return output_message_batter, similarity_table_batter, percentile_bar_chart_batter, difference_from_avg_chart_batter, euclidean_distance_scatter_batter

@app.callback(
    [Output('pitcher-output-container', 'children'),
     Output('pitcher-similarity-graph-container', 'children'),
     Output('pitcher-percentile-bar-chart-container', 'children'),
     Output('pitcher-difference-from-avg-chart-container', 'children'),
     Output('pitcher-euclidean-distance-scatter-container', 'children')],
    [Input('pitcher-dropdown', 'value')]
)
def update_pitcher_output(pitcher_name_id_input):
    if pitcher_name_id_input is None:
        return "선수, 시즌을 선택해주세요 (MLB ID 6자리로도 검색 가능합니다)", no_update, no_update, no_update, no_update

    # 선택된 선수에 대한 유사도 분석 수행
    similar_pitchers = calculate_pitcher_similarity(pitcher_name_id_input)

    # 결과가 없을 경우 처리
    if similar_pitchers is None or similar_pitchers.empty:
        return f"선수 {pitcher_name_id_input}의 데이터를 찾을 수 없습니다.", no_update, no_update, no_update, no_update

    # 유클리디안 거리의 최댓값 계산
    max_distance = similar_pitchers['euclidean_distance'].max()

    # 필터링 없이 전체 데이터 사용
    filtered_pitchers = similar_pitchers

    # 상위 5명의 유사도 높은 선수들 선택
    similar_pitchers_top5 = filtered_pitchers.sort_values(by='euclidean_distance', ascending=True).head(6)

    # 테이블 생성
    similarity_table_pitcher = dcc.Graph(figure=display_table_for_pitchers(similar_pitchers_top5))

    # 백분위수 막대 그래프 생성
    percentile_bar_chart_pitcher = dcc.Graph(figure=plot_percentile_bar_chart_for_pitchers(similar_pitchers_top5))

    # 차이 그래프 생성
    first_pitcher_row = similar_pitchers_top5.iloc[0]
    pitcher_name = first_pitcher_row['pitcher_name_new']
    game_year_pitcher = int(first_pitcher_row['game_year'])
    difference_from_avg_chart_pitcher = dcc.Graph(figure=plot_difference_from_avg_for_pitchers(similar_pitchers_top5, df_swing_table_all, pitcher_name, game_year_pitcher))

    # 유클리디안 거리 차트 생성
    euclidean_distance_scatter_pitcher = dcc.Graph(figure=plot_euclidean_distance_scatter_for_pitchers(filtered_pitchers, [0, max_distance]))

    # 결과 출력 메시지 생성
    output_message_pitcher = f"{game_year_pitcher} 시즌 {pitcher_name}의 유사도 분석 결과입니다."

    # 검색 후에만 그래프를 렌더링
    return output_message_pitcher, similarity_table_pitcher, percentile_bar_chart_pitcher, difference_from_avg_chart_pitcher, euclidean_distance_scatter_pitcher

#-----------------------------------------------------------------------------------------------------------------------------

# 유클리드 거리 상위 5명 추출 함수(타자)
def display_table_for_batters(similar_batters_top5):
    # 추출할 컬럼 리스트
    columns_to_display = [
        'game_year', 'batter_name_new', 'euclidean_distance', 'out_zone_swing%', 'zone_swing%', 'swing%', 'zone%',
        'out_zone_contact%', 'zone_contact%', 'contact%', 'first_pitch_swing%', 'swstr%', 'cstr%', 'csw%'
    ]

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
        header=dict(values=formatted_columns, fill_color='paleturquoise', align='center', height=40, font=dict(size=14)),
        cells=dict(values=[table_df[col] for col in table_df.columns], fill_color='white', align='center', height=40, font=dict(size=12))
    )])

    fig.update_layout(
        margin=dict(l=0, r=0, t=100, b=0),
        height=500,
        title=f"{game_year} Season {batter_name}<br>Top 5 Plate Discipline Similar Batters",
        title_x=0,
        title_xanchor='left',
        title_font=dict(size=20, family='Arial', color='black'),
        autosize=True,
        # 스크롤을 가능하게 하고 폭을 조절하는 부분
        xaxis=dict(
            automargin=True,
            zeroline=False,
            showgrid=False
        ),
    )

    return fig

# 유클리드 거리 상위 5명 추출 함수(투수)
def display_table_for_pitchers(similar_pitchers_top5):
    # 추출할 컬럼 리스트
    columns_to_display = [
        'game_year', 'pitcher_name_new', 'euclidean_distance', 'out_zone_swing%', 'zone_swing%', 'swing%', 'zone%',
        'out_zone_contact%', 'zone_contact%', 'contact%', 'first_pitch_strike%', 'swstr%', 'cstr%', 'csw%'
    ]

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
        title_font=dict(size=20, family='Arial', color='black')
    )

    return fig

#-----------------------------------------------------------------------------------------------------------------------------

def plot_percentile_bar_chart_for_batters(similar_batters):
    # 5개의 백분위수 컬럼 리스트
    batter_percentile_columns = [
        'contact%_std_percentile', 'zone_contact%_std_percentile',
        'zone_swing%_std_percentile', 'out_zone_swing%_std_percentile', 'swstr%_std_percentile'
    ]

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
    actual_columns = ['contact%', 'zone_contact%', 'zone_swing%', 'out_zone_swing%', 'swstr%']
    actual_values = similar_batters.iloc[0][actual_columns].values

    # 선수의 PA가 기준 미만이면 투명도를 조정하고 텍스트 출력 안 함
    if target_pa < pa_threshold:
        percentile_values = similar_batters.iloc[0][batter_percentile_columns].values
        text_values = ''  # 출력 안 함
        opacity_value = 0.3  # 투명도를 연하게 설정
        # hovertext에 백분위수 포함
        hovertext_values = [
            f"측정값 : {actual_values[i]:.1f}%, 백분위수: {int(percentile_values[i])}"
            for i in range(len(batter_percentile_columns))
        ]
    else:
        percentile_values = similar_batters.iloc[0][batter_percentile_columns].values
        text_values = [f"{int(value)}" for value in percentile_values]  # 실제 백분위수 값
        opacity_value = 1.0  # 투명도를 기본 값으로 설정
        # hovertext에서 백분위수는 제외하고 측정값만 표시
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
            showgrid=False
        )
    )

    return fig

def plot_percentile_bar_chart_for_pitchers(similar_pitchers):
    # 9개의 백분위수 컬럼 리스트
    pitcher_percentile_columns = [
        'contact%_std_percentile', 'zone_contact%_std_percentile', 'out_zone_contact%_std_percentile',
        'zone_swing%_std_percentile', 'out_zone_swing%_std_percentile', 'first_pitch_strike%_std_percentile',
        'swstr%_std_percentile', 'cstr%_std_percentile', 'csw%_std_percentile'
    ]

    # 선수의 PA와 game_year 가져옴
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
    actual_columns = ['contact%', 'zone_contact%', 'out_zone_contact%',
                      'zone_swing%', 'out_zone_swing%', 'first_pitch_strike%',
                      'swstr%', 'cstr%', 'csw%']
    actual_values = similar_pitchers.iloc[0][actual_columns].values

    # 선수의 PA가 기준 미만이면 투명도를 조정하고 텍스트 출력 안 함
    if target_pa < pa_threshold:
        percentile_values = similar_pitchers.iloc[0][pitcher_percentile_columns].values
        text_values = ''  # 출력 안 함
        opacity_value = 0.3  # 투명도를 연하게 설정
        # hovertext에 백분위수 포함
        hovertext_values = [
            f"측정값 : {actual_values[i]:.1f}%, 백분위수 : {int(percentile_values[i])}"
            for i in range(len(pitcher_percentile_columns))
        ]
    else:
        percentile_values = similar_pitchers.iloc[0][pitcher_percentile_columns].values
        text_values = [f"{int(value)}" for value in percentile_values]  # 실제 백분위수 값
        opacity_value = 1.0  # 투명도를 기본 값으로 설정
        # hovertext에서 백분위수는 제외하고 측정값만 표시
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
        height=450,  # 그래프의 높이 설정
        template="plotly_white",
        xaxis=dict(range=[0, 105]),  # x축 범위를 0~105로 설정
        yaxis=dict(
            automargin=True,  # 레이블 간격 조정
            showgrid=False,  # y축에 격자선 제거
            autorange='reversed'
        )
    )

    return fig

#-----------------------------------------------------------------------------------------------------------------------------

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
        for i, y_label in enumerate(y_labels)
    ]

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
        template="plotly_white"
    )

    return fig

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
        for i, y_label in enumerate(y_labels)
    ]

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
        template="plotly_white"
    )

    return fig

#-----------------------------------------------------------------------------------------------------------------------------

def plot_euclidean_distance_scatter_for_batters(similar_batters, slider_range):
    # slider_range로 전달받은 범위로 데이터 필터링
    low, high = slider_range
    filtered_df = similar_batters.iloc[1:][(similar_batters['euclidean_distance'] >= low) & 
                                   (similar_batters['euclidean_distance'] <= high)]

    # 점 그래프 생성
    fig = go.Figure(data=go.Scatter(
        x=filtered_df['euclidean_distance'],
        y=filtered_df['pa'],
        mode='markers',
        marker=dict(
            size=8,
            color=filtered_df['euclidean_distance'],  # euclidean_distance에 따라 색상 지정
            colorscale='Agsunset',
            showscale=False  # 색상 범례 숨김
        ),
        name="Player Data",  # 범례에 표시할 이름
        hovertext=[
            f"{row['batter_name_new']} ({int(row['game_year'])})<br>"
            f"PA : {row['pa']}<br>"
            f"Euclidean Distance : {row['euclidean_distance']:.3f}<br>"
            f"Out Zone Swing% : {row['out_zone_swing%']:.1f}%<br>"
            f"Zone Swing% : {row['zone_swing%']:.1f}%<br>"
            f"Swing% : {row['swing%']:.1f}%<br>"
            f"Zone% : {row['zone%']:.1f}%<br>"
            f"Out Zone Contact% : {row['out_zone_contact%']:.1f}%<br>"
            f"Zone Contact% : {row['zone_contact%']:.1f}%<br>"
            f"Contact% : {row['contact%']:.1f}%<br>"
            f"First Pitch Strike% : {row['first_pitch_strike%']:.1f}%<br>"
            f"First Pitch Swing% : {row['first_pitch_swing%']:.1f}%<br>"
            f"SwStr% : {row['swstr%']:.1f}%<br>"
            f"CStr% : {row['cstr%']:.1f}%<br>"
            f"CSW% : {row['csw%']:.1f}%"
            for _, row in filtered_df.iterrows()
        ],
        hoverinfo='text'
    ))

    # 유클리디안 거리의 최대값 계산
    max_distance = math.ceil(similar_batters['euclidean_distance'].max())

    # 슬라이더 첫 번째 값을 0 초과 값으로 설정
    slider_values = np.linspace(0.01, max_distance, 100)  # 0.01부터 시작

    # 레이아웃 업데이트
    fig.update_layout(
        sliders=[{
            'yanchor': 'top',
            'xanchor': 'left',
            'pad': {'b': 5},
            'steps': [
                {'method': 'relayout', 'label': '',
                 'args': [{'xaxis.range': [0, val]}]}  # 항상 x축의 최소값을 0으로 설정
                for val in slider_values  # 100 단계로 슬라이더 설정
            ],
            'tickwidth': 0
        }],
        title="Compare All Similar Batters",
        title_x=0,  # 제목의 x축 위치 설정 (0 = 왼쪽 정렬)
        title_xanchor='left',  # 제목을 왼쪽 정렬로 설정
        title_font=dict(size=20, family='Arial', color='black'),
        xaxis=dict(rangeslider_visible=True),
        yaxis=dict(title='PA'),
        height=650,
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=40),
        legend=dict(
            title="Game year"
        )
    )

    return fig

def plot_euclidean_distance_scatter_for_pitchers(similar_pitchers, slider_range):
    # slider_range로 전달받은 범위로 데이터 필터링
    low, high = slider_range
    filtered_df = similar_pitchers.iloc[1:][(similar_pitchers['euclidean_distance'] >= low) & 
                                   (similar_pitchers['euclidean_distance'] <= high)]

    # 점 그래프 생성
    fig = go.Figure(data=go.Scatter(
        x=filtered_df['euclidean_distance'],
        y=filtered_df['pa'],
        mode='markers',
        marker=dict(
            size=8,
            color=filtered_df['euclidean_distance'],  # euclidean_distance에 따라 색상 지정
            colorscale='Agsunset',
            showscale=False  # 색상 범례 숨김
        ),
        name="Player Data",  # 범례에 표시할 이름
        hovertext=[
            f"{row['pitcher_name_new']} ({int(row['game_year'])})<br>"
            f"PA : {row['pa']}<br>"
            f"Euclidean Distance : {row['euclidean_distance']:.3f}<br>"
            f"Out Zone Swing% : {row['out_zone_swing%']:.1f}%<br>"
            f"Zone Swing% : {row['zone_swing%']:.1f}%<br>"
            f"Swing% : {row['swing%']:.1f}%<br>"
            f"Zone% : {row['zone%']:.1f}%<br>"
            f"Out Zone Contact% : {row['out_zone_contact%']:.1f}%<br>"
            f"Zone Contact% : {row['zone_contact%']:.1f}%<br>"
            f"Contact% : {row['contact%']:.1f}%<br>"
            f"First Pitch Strike% : {row['first_pitch_strike%']:.1f}%<br>"
            f"First Pitch Swing% : {row['first_pitch_swing%']:.1f}%<br>"
            f"SwStr% : {row['swstr%']:.1f}%<br>"
            f"CStr% : {row['cstr%']:.1f}%<br>"
            f"CSW% : {row['csw%']:.1f}%"
            for _, row in filtered_df.iterrows()
        ],
        hoverinfo='text'
    ))

    # 유클리디안 거리의 최대값 계산
    max_distance = math.ceil(similar_pitchers['euclidean_distance'].max())

    # 슬라이더 첫 번째 값을 0 초과 값으로 설정
    slider_values = np.linspace(0.01, max_distance, 100)  # 0.01부터 시작

    # 레이아웃 업데이트
    fig.update_layout(
        sliders=[{
            'yanchor': 'top',
            'xanchor': 'left',
            'pad': {'b': 5},
            'steps': [
                {'method': 'relayout', 'label': '',
                 'args': [{'xaxis.range': [0, val]}]}  # 항상 x축의 최소값을 0으로 설정
                for val in slider_values  # 100 단계로 슬라이더 설정
            ],
            'tickwidth': 0
        }],
        title="Compare All Similar Pitchers",
        title_x=0,  # 제목의 x축 위치 설정 (0 = 왼쪽 정렬)
        title_xanchor='left',  # 제목을 왼쪽 정렬로 설정
        title_font=dict(size=20, family='Arial', color='black'),
        xaxis=dict(rangeslider_visible=True),
        yaxis=dict(title='PA'),
        height=650,
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=40),
        legend=dict(
            title="Game year"
        )
    )

    return fig

# 애플리케이션 실행
if __name__ == '__main__':
    app.run_server(debug=True)
#-----------------------------------------------------------------------------------------------------------------------------