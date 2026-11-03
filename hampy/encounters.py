import numpy as np

def get_enc_dates(enc_num):
    match enc_num:
        case 'E01': return np.datetime64('2018-11-06T03:27')

        case 'E02': return np.datetime64('2019-04-04T22:40')

        case 'E03': return np.datetime64('2019-09-01T17:50')

        case 'E04': return np.datetime64('2020-01-29T09:37')

        case 'E05': return np.datetime64('2020-06-07T08:23')

        case 'E06': return np.datetime64('2020-09-27T00:00')

        case 'E07': return np.datetime64('2021-01-17T00:00')

        case 'E08': return np.datetime64('2021-04-28T00:00')

        case 'E09': return np.datetime64('2021-08-09T00:00')

        case 'E10': return np.datetime64('2021-11-21T00:00')

        case 'E11': return np.datetime64('2022-02-25T00:00')

        case 'E12': return np.datetime64('2022-06-01T00:00')

        case 'E13': return np.datetime64('2022-09-06T00:00')

        case 'E14': return np.datetime64('2022-12-11T00:00')

        case 'E15': return np.datetime64('2023-03-17T00:00')

        case 'E16': return np.datetime64('2023-06-22T00:00')

        case 'E17': return np.datetime64('2023-09-27T00:00')

        case 'E18': return np.datetime64('2023-12-29T00:00')

        case 'E19': return np.datetime64('2024-03-30T00:00')

        case 'E20': return np.datetime64('2024-06-30T00:00')

        case 'E21': return np.datetime64('2024-09-30T00:00')

        case 'E22': return np.datetime64('2024-12-24T00:00')

        case 'E23': return np.datetime64('2025-03-22T00:00')

        case 'E24': return np.datetime64('2025-06-19T00:00')

        case 'E25': return np.datetime64('2025-09-15T00:00')

        case 'E26': return np.datetime64('2025-12-13T00:00')