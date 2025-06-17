df['column'] = df['column'].str.strip("123./@#$%^&*")
this would remove 1,2,3,.,/,@,#,$,%,^,&,* from the entire column


if we have a column in which we want just numbers and not anything else so we follow the following script
df['phone'] = df['phone'].str.replace('[^a-zA-Z0-9]', '')


