import keras
from keras.layers import Embedding, Masking, GRU, Flatten, RepeatVector, Concatenate, TimeDistributed, Dense, \
    Dropout

company_code = keras.Input(
    shape=(1,),
    name='company_code',
    dtype='int32'
)

company_code_embedding = Embedding(
    input_dim=200, output_dim=49, name='company_code_embedding'
)(company_code)

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

# ay_input = keras.Input(
#     shape=(9,2),
#     name='ay_input'
# )

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

paid_output = TimeDistributed(Dense(units=64, activation='relu'))(decoded)
paid_output = TimeDistributed(Dropout(rate=0.2))(paid_output)
paid_output = TimeDistributed(Dense(units=1, activation='relu'), name='Paid_Output')(paid_output)


def get_deep_triangle():
    return keras.Model(
        inputs=[company_code, loss_paid, case_reserves],
        outputs=[paid_output, case_reserves_output]
    )