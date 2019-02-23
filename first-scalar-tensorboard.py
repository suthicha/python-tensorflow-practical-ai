import tensorflow as tf


# clear define variables and operations of the previous cell.
tf.reset_default_graph()

# create a scalar variable.
x_scalar = tf.get_variable('x_scalar', shape=[], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))

# step 1. create a scalar summary.
first_summary = tf.summary.scalar(name="MY_FIRST_SCALAR_SUMMARY", tensor=x_scalar)

init = tf.global_variables_initializer()

# launch the graph in a session.
with tf.Session() as session:
    # step 2. create the writer inside the session.
    writer = tf.summary.FileWriter('./graphs/scalar_summary', session.graph)

    for step in range(100):
        # loop over several init of the variables.
        session.run(init)

        # step 3. evaluate the scalar summary.
        summary = session.run(first_summary)

        # step 4. add the summary to the writer (i.e. to the event file)
        writer.add_summary(summary, step)

    print('Done with writing the scalar summary')

writer.close()
