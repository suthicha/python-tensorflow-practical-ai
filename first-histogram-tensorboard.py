import tensorflow as tf

# clear define variables and operations of the previous cell.
tf.reset_default_graph()

# create variables
x_scalar = tf.get_variable('x_scalar', shape=[],
                           initializer=tf.truncated_normal_initializer(mean=0, stddev=1))

x_matrix = tf.get_variable('x_matrix', shape=[30, 40],
                           initializer=tf.truncated_normal_initializer(mean=0, stddev=1))

# step 1.1: create the scalar summary.
scalar_summary = tf.summary.scalar(name='MY_FIRST_SCALAR_SUMMARY', tensor=x_scalar)

# step 1.2: create the histogram summary for the non-scalar (i.e. 2D or matrix) tensor
histogram_summary = tf.summary.histogram('MY_FIRST_HISTOGRAM_SUMMARY', values=x_matrix)

init = tf.global_variables_initializer()

# launch the graph in a session
with tf.Session() as session:
    # step 2: create the writer inside the session.
    writer = tf.summary.FileWriter('./graphs/histogram_summary', session.graph)

    for step in range(100):
        # loop over several init of the variables.
        session.run(init)

        # step 3: evaluate the scalar summary.
        summary1, summary2 = session.run([scalar_summary, histogram_summary])

        # step 4: add the summary to the writer (i.e. the event file)
        writer.add_summary(summary1, step)
        writer.add_summary(summary2, step)

    print('Done with writing the histogram summary')

writer.close()
