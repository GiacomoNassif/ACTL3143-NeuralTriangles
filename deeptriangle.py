import keras
import numpy as np
from keras.layers import Embedding, Masking, GRU, Flatten, RepeatVector, Concatenate, TimeDistributed, Dense, \
    Dropout, IntegerLookup, LSTM


def get_deep_triangle(company_codes: np.ndarray):
    company_code = keras.Input(
        shape=(1,),
        name='company_code',
        dtype='int32'
    )

    company_code_mapping = IntegerLookup()
    company_code_mapping.adapt(company_codes)

    company_code_mapped = company_code_mapping(company_code)

    company_code_embedding = Embedding(
        input_dim=200, output_dim=49, name='company_code_embedding'
    )(company_code_mapped)

    company_code_embedding = Flatten()(company_code_embedding)
    company_code_embedding = RepeatVector(9)(company_code_embedding)

    loss_paid = keras.Input(
        shape=(9, 1),
        name='Incremental_Loss_Paid_Loss_Ratio'
    )

    case_reserves = keras.Input(
        shape=(9, 1),
        name='Case_Reserves_Loss_Ratio'
    )

    ay_input = Concatenate(axis=2)([loss_paid, case_reserves])

    encoded = Masking(mask_value=-99)(ay_input)
    encoded = GRU(units=128, dropout=0.2, recurrent_dropout=0.2)(encoded)

    decoded = RepeatVector(n=9)(encoded)
    decoded = GRU(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(decoded)

    history_and_company_code = Concatenate(axis=2)([decoded, company_code_embedding])

    case_reserves_output = TimeDistributed(Dense(units=64, activation='relu'))(history_and_company_code)
    case_reserves_output = TimeDistributed(Dropout(rate=0.2))(case_reserves_output)
    case_reserves_output = TimeDistributed(
        Dense(units=1, activation='relu'),
        name='Case_Reserves_Output')(
        case_reserves_output
    )

    paid_output = TimeDistributed(Dense(units=64, activation='relu'))(history_and_company_code)
    paid_output = TimeDistributed(Dropout(rate=0.2))(paid_output)
    paid_output = TimeDistributed(Dense(units=1, activation='relu'), name='Paid_Output')(paid_output)

    return keras.Model(
        inputs=[company_code, loss_paid, case_reserves],
        outputs=[paid_output, case_reserves_output]
    )



def get_deep_triangle2(company_codes: np.ndarray, number_of_companies_to_rep):
    company_code = keras.Input(
        shape=(1,),
        name='company_code',
        dtype='int32'
    )

    company_code_mapping = IntegerLookup()
    company_code_mapping.adapt(company_codes)

    company_code_mapped = company_code_mapping(company_code)

    company_code_embedding = Embedding(
        input_dim=200, output_dim=number_of_companies_to_rep, name='company_code_embedding'
    )(company_code_mapped)

    company_code_embedding = Flatten()(company_code_embedding)
    company_code_embedding = RepeatVector(9)(company_code_embedding)

    loss_paid = keras.Input(
        shape=(9, 1),
        name='Incremental_Loss_Paid_Loss_Ratio'
    )

    case_reserves = keras.Input(
        shape=(9, 1),
        name='Case_Reserves_Loss_Ratio'
    )

    ay_input = Concatenate(axis=2)([loss_paid, case_reserves])

    encoded = Masking(mask_value=-99)(ay_input)
    encoded = LSTM(units=128, dropout=0.2, recurrent_dropout=0.2)(encoded)

    decoded = RepeatVector(n=9)(encoded)
    decoded = LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(decoded)

    history_and_company_code = Concatenate(axis=2)([decoded, company_code_embedding])

    case_reserves_output = TimeDistributed(Dense(units=64, activation='relu'))(history_and_company_code)
    case_reserves_output = TimeDistributed(Dropout(rate=0.2))(case_reserves_output)
    case_reserves_output = TimeDistributed(
        Dense(units=1, activation='relu'),
        name='Case_Reserves_Output')(
        case_reserves_output
    )

    paid_output = TimeDistributed(Dense(units=64, activation='relu'))(history_and_company_code)
    paid_output = TimeDistributed(Dropout(rate=0.2))(paid_output)
    paid_output = TimeDistributed(Dense(units=1, activation='relu'), name='Paid_Output')(paid_output)

    return keras.Model(
        inputs=[company_code, loss_paid, case_reserves],
        outputs=[paid_output, case_reserves_output]
    )

import keras.backend as K

"""
Source: https://stackoverflow.com/a/70126561/10916209
"""


def get_loss(mask_value):
    mask_value = K.variable(mask_value)

    def masked_loss(yTrue, yPred):
        # find which values in yTrue (target) are the mask value
        isMask = K.equal(yTrue, mask_value)  # true for all mask values

        # transform to float (0 or 1) and invert
        isMask = K.cast(isMask, dtype=K.floatx())
        isMask = 1 - isMask  # now mask values are zero, and others are 1

        # multiply this by the inputs:
        # maybe you might need K.expand_dims(isMask) to add the extra dimension removed by K.all
        yTrue = yTrue * isMask
        yPred = yPred * isMask

        # perform a root mean square error, whereas the mean is in respect to the mask
        mean_loss = K.sum(K.square(yPred - yTrue)) / K.sum(isMask)

        return mean_loss

    return masked_loss
