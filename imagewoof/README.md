# Prelimiries

`python setup.py` - download dataset and create directories 'teachers' and 'students'

# Training teachers

`python train_teacher.py <teacher_num>` - train teacher with <teacher_num> number. Save the model and training/validating results to directorie 'teachers'
`python train__long_teacher.py <teacher_num>` - train long teacher (3 * epochs) with <teacher_num> number. Save the model and training/validating results to directorie 'teachers'

# Training student

`python train_student.py <student_num> <teacher_num1> <teacher_num2>` - train student using teachers with numbers <teacher_num1> and <teacher_num2>. Model and training/testing results saved to directorie 'students'.
