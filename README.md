# RNN Tutorial: A Binary Addition Example
This is an implementation of a simple RNN modeling the Binary Addition . 

# Usage

```console
>> from RNNBinaryAdd import RNNBinaryAdd
>> rnn = RNNBinaryAdd(hid_dim=16)
>> rnn.rnnBinaryAdd(34,4)
The true answer is: 38
The predication is: 0.0
>> rnn.rnnBinaryAdd(34,46)
The true answer is: 80
The predication is: 0.0
>> rnn.rnnBinaryAdd(34,57)
The true answer is: 91
The predication is: 0.0
>> rnn.train(alpha = 0.1, epoch=1, train_sample= 10000)
target     :[1, 0, 1, 0, 0, 0, 1, 0]
predict rnd:[1, 0, 0, 0, 0, 0, 0, 0]
predict raw:[0.52, 0.27, 0.35, 0.43, 0.30, 0.35, 0.39, 0.36]
Overall abs error: 3.45638663235765
target     :[1, 1, 1, 1, 1, 1, 0, 0]
predict rnd:[1, 1, 1, 1, 1, 1, 1, 1]
predict raw:[0.54, 0.58, 0.65, 0.60, 0.60, 0.55, 0.58, 0.53]
Overall abs error: 3.589346200756878
target     :[0, 0, 0, 0, 0, 1, 0, 1]
predict rnd:[0, 0, 0, 1, 1, 1, 1, 0]
predict raw:[0.41, 0.40, 0.44, 0.51, 0.54, 0.53, 0.55, 0.46]
Overall abs error: 3.86849928221804
target     :[1, 0, 1, 1, 0, 0, 1, 0]
predict rnd:[0, 1, 1, 0, 0, 0, 0, 0]
predict raw:[0.46, 0.50, 0.54, 0.48, 0.47, 0.42, 0.43, 0.46]
Overall abs error: 3.9486405735778365
target     :[0, 1, 0, 0, 1, 0, 1, 0]
predict rnd:[0, 1, 1, 1, 0, 0, 0, 0]
predict raw:[0.42, 0.56, 0.61, 0.52, 0.41, 0.31, 0.39, 0.37]
Overall abs error: 3.872696947354195
target     :[0, 1, 0, 0, 0, 0, 1, 1]
predict rnd:[0, 1, 0, 0, 0, 1, 0, 1]
predict raw:[0.40, 0.64, 0.34, 0.19, 0.35, 0.51, 0.43, 0.67]
Overall abs error: 3.0494538867422967
target     :[1, 0, 0, 0, 1, 0, 1, 0]
predict rnd:[1, 1, 0, 0, 0, 0, 1, 0]
predict raw:[0.65, 0.58, 0.17, 0.08, 0.43, 0.20, 0.64, 0.31]
Overall abs error: 2.6261118673085013
target     :[1, 0, 0, 0, 0, 0, 0, 1]
predict rnd:[1, 0, 0, 0, 1, 1, 1, 1]
predict raw:[0.78, 0.39, 0.30, 0.45, 0.55, 0.57, 0.58, 0.60]
Overall abs error: 3.460563199897909
target     :[0, 0, 0, 1, 1, 1, 0, 0]
predict rnd:[0, 0, 0, 1, 1, 1, 0, 0]
predict raw:[0.15, 0.36, 0.48, 0.56, 0.89, 0.71, 0.26, 0.02]
Overall abs error: 2.1150181525645877
target     :[0, 1, 1, 1, 0, 0, 0, 0]
predict rnd:[0, 1, 1, 1, 0, 0, 0, 0]
predict raw:[0.17, 0.61, 0.79, 0.88, 0.06, 0.01, 0.01, 0.01]
Overall abs error: 0.9742302539677573
>> rnn.rnnBinaryAdd(34,4)
The true answer is: 38
The predication is: 38.0
>> rnn.rnnBinaryAdd(34,46)
The true answer is: 80
The predication is: 80.0
>> rnn.rnnBinaryAdd(34,57)
The true answer is: 91
The predication is: 95.0
```

Make another run with epoch = 2

```console
>> rnn.reset()
>> rnn.rnnBinaryAdd(34,57)
The true answer is: 91
The predication is: 0.0
>> rnn.train(alpha = 0.1, epoch=2, train_sample= 10000)
target     :[0, 1, 0, 0, 1, 0, 1, 0]
predict rnd:[0, 0, 0, 0, 0, 0, 0, 0]
predict raw:[0.27, 0.24, 0.23, 0.28, 0.23, 0.26, 0.20, 0.25]
Overall abs error: 3.622042121762073
target     :[1, 0, 0, 1, 1, 1, 1, 0]
predict rnd:[0, 0, 0, 0, 0, 1, 0, 0]
predict raw:[0.46, 0.50, 0.44, 0.49, 0.48, 0.53, 0.47, 0.50]
Overall abs error: 4.006820669200654
target     :[0, 1, 0, 0, 0, 1, 1, 0]
predict rnd:[0, 0, 0, 1, 0, 0, 1, 0]
predict raw:[0.47, 0.49, 0.49, 0.56, 0.47, 0.50, 0.54, 0.45]
Overall abs error: 3.9012467166997262
target     :[1, 1, 0, 0, 1, 0, 0, 1]
predict rnd:[0, 1, 0, 1, 0, 0, 0, 0]
predict raw:[0.49, 0.50, 0.49, 0.53, 0.44, 0.39, 0.42, 0.50]
Overall abs error: 3.8890573050898896
target     :[1, 1, 1, 0, 1, 1, 0, 1]
predict rnd:[1, 1, 1, 1, 1, 1, 1, 1]
predict raw:[0.58, 0.66, 0.63, 0.57, 0.54, 0.53, 0.58, 0.62]
Overall abs error: 3.597525271705182
target     :[0, 1, 0, 1, 1, 1, 0, 1]
predict rnd:[0, 1, 1, 1, 1, 0, 0, 1]
predict raw:[0.41, 0.57, 0.61, 0.52, 0.55, 0.49, 0.44, 0.58]
Overall abs error: 3.7634530758909004
target     :[0, 0, 1, 1, 0, 0, 0, 0]
predict rnd:[0, 0, 1, 1, 0, 0, 0, 0]
predict raw:[0.37, 0.27, 0.50, 0.64, 0.49, 0.23, 0.13, 0.11]
Overall abs error: 2.45758271705314
target     :[1, 1, 1, 0, 0, 1, 0, 1]
predict rnd:[1, 1, 1, 0, 1, 0, 1, 1]
predict raw:[0.65, 0.64, 0.63, 0.36, 0.59, 0.46, 0.60, 0.74]
Overall abs error: 3.416242383752707
target     :[1, 1, 0, 0, 1, 1, 1, 0]
predict rnd:[1, 1, 1, 0, 1, 1, 1, 1]
predict raw:[0.71, 0.59, 0.69, 0.03, 0.87, 0.71, 0.55, 0.81]
Overall abs error: 3.0990000569662275
target     :[1, 1, 1, 1, 0, 1, 1, 0]
predict rnd:[1, 1, 1, 1, 0, 1, 1, 0]
predict raw:[0.84, 0.58, 0.64, 0.53, 0.45, 0.88, 0.90, 0.02]
Overall abs error: 2.0854572022296542
target     :[0, 1, 0, 0, 1, 0, 1, 0]
predict rnd:[0, 1, 0, 0, 1, 0, 1, 0]
predict raw:[0.17, 0.96, 0.06, 0.22, 0.81, 0.26, 0.91, 0.00]
Overall abs error: 1.02609407109118
target     :[1, 0, 0, 1, 1, 1, 1, 0]
predict rnd:[1, 0, 0, 1, 1, 1, 1, 0]
predict raw:[0.86, 0.02, 0.00, 0.98, 0.98, 0.96, 0.89, 0.05]
Overall abs error: 0.39608604303494366
target     :[0, 1, 0, 0, 0, 1, 1, 0]
predict rnd:[0, 1, 0, 0, 0, 1, 1, 0]
predict raw:[0.09, 0.98, 0.04, 0.07, 0.10, 0.90, 0.91, 0.00]
Overall abs error: 0.5017257128199452
target     :[1, 1, 0, 0, 1, 0, 0, 1]
predict rnd:[1, 1, 0, 0, 1, 0, 0, 1]
predict raw:[0.92, 0.97, 0.08, 0.03, 0.96, 0.00, 0.03, 0.97]
Overall abs error: 0.32673608922856423
target     :[1, 1, 1, 0, 1, 1, 0, 1]
predict rnd:[1, 1, 1, 0, 1, 1, 0, 1]
predict raw:[0.93, 0.99, 0.97, 0.01, 0.99, 0.99, 0.02, 0.96]
Overall abs error: 0.1943086719562322
target     :[0, 1, 0, 1, 1, 1, 0, 1]
predict rnd:[0, 1, 0, 1, 1, 1, 0, 1]
predict raw:[0.07, 0.98, 0.09, 0.96, 0.94, 0.99, 0.02, 0.98]
Overall abs error: 0.3238500101937706
target     :[0, 0, 1, 1, 0, 0, 0, 0]
predict rnd:[0, 0, 1, 1, 0, 0, 0, 0]
predict raw:[0.06, 0.00, 0.99, 1.00, 0.00, 0.00, 0.00, 0.00]
Overall abs error: 0.08155365897432774
target     :[1, 1, 1, 0, 0, 1, 0, 1]
predict rnd:[1, 1, 1, 0, 0, 1, 0, 1]
predict raw:[0.94, 0.98, 0.93, 0.07, 0.01, 0.98, 0.10, 0.91]
Overall abs error: 0.4426441360446768
target     :[1, 1, 0, 0, 1, 1, 1, 0]
predict rnd:[1, 1, 0, 0, 1, 1, 1, 0]
predict raw:[0.95, 0.99, 0.00, 0.00, 0.99, 1.00, 0.99, 0.00]
Overall abs error: 0.08672934758636608
target     :[1, 1, 1, 1, 0, 1, 1, 0]
predict rnd:[1, 1, 1, 1, 0, 1, 1, 0]
predict raw:[0.95, 0.99, 0.98, 0.99, 0.02, 0.98, 1.00, 0.00]
Overall abs error: 0.13664365313082696
>> rnn.rnnBinaryAdd(34,57)
The true answer is: 91
The predication is: 91.0
```
