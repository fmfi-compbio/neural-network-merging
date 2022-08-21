# Prelimiries

`python setup.py` - create directories 'teachers' and 'students'

# Training teachers

`python train_teacher.py ` - train 150 teachers. Save the models and training/validating results to directory 'teachers'\
`python train__long_teacher.py` - train 50 long teacher (3 * epochs). Save the models and training/validating results to directory 'teachers'

# Training student

`python train_student.py` - train 50 students using first 100 teachers. Models and training/testing results are saved to directory 'students'.
