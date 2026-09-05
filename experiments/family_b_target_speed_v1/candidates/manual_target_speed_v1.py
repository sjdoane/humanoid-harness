CANDIDATE_ID = "manual_target_speed_v1"
VERSION = 1


def task_term(x):
    return 1.25 * (
        1.0
        - min(
            1.0,
            abs(x.com_x_velocity_m_s - x.target_speed_m_s) / x.target_speed_m_s,
        )
    )
