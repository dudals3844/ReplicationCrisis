import pandas as pd 

settings = {
    "seed": 1,
    "start_date": pd.Timestamp("1925-12-31"),
    # 'end_date': datetime.date(2022, 12, 31),  # end_date는 반드시 Last_CRSP_UPDATE 이하이어야 합니다
    "end_date": pd.Timestamp(
        "2022-12-31"
    ),  # end_date는 반드시 Last_CRSP_UPDATE 이하이어야 합니다
    "country_excl": ["ZWE", "VEN"],  # 데이터 문제로 인해 제외된 국가들
    "weighting": {  # 사용할 가중치 방식 (선택지: "ew", "vw", "vw_cap")
        "us": "vw_cap",
        "global_ex_us": "vw_cap",
    },
    "n_stocks_min": 5,  # 각 포트폴리오 측면에서 최소한의 주식 수
    "months_min": 5 * 12,  # 팩터가 포함되기 위해 필요한 최소한의 관측 기간
    "country_weighting": "market_cap",  # 국가 가중치를 부여하는 방법 ("market_cap", "stocks", "ew")
    "countries_min": 3,  # 지역 포트폴리오에 필요한 최소 국가 수
    "clusters": "hcl",  # 사용할 클러스터링 방법 (선택지: "manual", "hcl")
    "hcl": {
        "ret_type": "alpha",  # 클러스터링에 사용할 수익 유형 (선택지: "raw", "alpha")
        "cor_method": "pearson",  # 거리 계산에 사용할 상관관계 방법
        "linkage": "ward.D",  # 클러스터링에 사용할 연결 방법
        "k": 13,  # 색칠할 클러스터 수
        "region": "us",  # 클러스터링에 사용할 지역
        "start_year": 1975,  # 클러스터 데이터 시작 연도
    },
    "eb": {
        "scale_alpha": True,
        "overlapping": False,
        "min_obs": 5 * 12,
        "fix_alpha": True,
        "bs_cov": True,
        "shrinkage": 0,
        "cor_type": "block_clusters",
        "bs_samples": 10000,  # 부트스트랩 샘플 수 (논문에서는 10000으로 설정)
    },
    "tpf": {
        "start": {
            "world": pd.Timestamp("1952-1-1"),
            "us": pd.Timestamp("1952-1-1"),
            "developed": pd.Timestamp("1987-1-1"),
            "emerging": pd.Timestamp("1994-1-1"),
            "size_grps": pd.Timestamp("1963-1-1"),  # 나노캡 시작에 의해 결정됨
        },
        "bs_samples": 10000,  # 부트스트랩 샘플 수 [논문에서는 10,000]
        "shorting": False,  # 공매도가 허용되어야 하는가?
    },
    "tpf_factors": {
        "region": "us",
        "orig_sig": True,  # 원래 중요한 팩터만 포함: True, 모두 포함: (True, False)
        "start": pd.Timestamp("1972-1-31"),
        "scale": True,  # 사후 변동성 10%에 맞춰 조정할 것인가?
        "k": 5,  # 교차 검증을 위한 폴드 수
    },
}