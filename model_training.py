from resources.data_scripts.Keras_data_manipulation import get_insurance_line_data
from deeptriangle import get_deep_triangle, get_loss, get_deep_triangle2
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

training_data, validation_data, test_data = get_insurance_line_data('Private passenger auto')

training_X = training_data[:3]
training_Y = training_data[3:]
val_X, val_Y = validation_data[:3], validation_data[3:]

deep_triangle = get_deep_triangle(training_data[0])
loss_func = get_loss(-99)

deep_triangle.compile(
    Adam(learning_rate=0.0005, amsgrad=True),
    [loss_func, loss_func],
    loss_weights=[0.5, 0.5]
)

es = EarlyStopping(min_delta=0.001, patience=200, mode='min', restore_best_weights=True)

deep_triangle.fit(
    training_X,
    training_Y,
    batch_size=2250,
    epochs=1000,
    callbacks=[es],
    validation_data=(val_X, val_Y)
)


test_X, test_Y = test_data[:3], test_data[3:]
test_X, test_Y = test_data[:3], test_data[3:]

def train_and_val_model(company_reals):
    deep_triangle = get_deep_triangle2(training_data[0], company_reals)
    loss_func = get_loss(-99)

    deep_triangle.compile(
        Adam(learning_rate=0.0005, amsgrad=True),
        [loss_func, loss_func],
        loss_weights=[0.5, 0.5]
    )

    es = EarlyStopping(min_delta=0.001, patience=200, mode='min', restore_best_weights=True)

    deep_triangle.fit(
        training_X,
        training_Y,
        batch_size=2250,
        epochs=20,
        callbacks=[es],
        validation_data=(val_X, val_Y),
        verbose=False
    )

    train_loss = deep_triangle.evaluate(test_X, test_Y)

    print(f'Hyper Parameter {company_reals}, Loss: {train_loss}')


hyper_paramers = [10, 50, 100]

for param in hyper_paramers:
    train_and_val_model(param)


deep_triangle.evaluate(test_X, test_Y)
