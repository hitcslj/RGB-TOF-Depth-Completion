### Building an evaluation program that works with CodaLab

This example uses python.

`evaluate.py` - is an example that checks that the submission data matches the truth data, which is "Hello World!"
`utils.py` - this is a file that provides basic functions for the evaluate.py script.
`metadata` - this is a file that lists the contents of the program.zip bundle for the CodaLab system.

Once these pieces are assembled they are packages as program.zip which CodaLab can then use to evaluate the submissions
for a competition.

command: python $program/evaluate.py 
description: Example competition evaluation program.