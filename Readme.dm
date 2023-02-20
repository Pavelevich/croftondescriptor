ImproveBeta1:
(interpolation)

Use descriptive variable names: Using descriptive names for variables and functions can make the code more readable and understandable. For example, 
instead of using "i" and "j" as variable names, use "startIndex" and "endIndex".

Avoid using magic numbers: Instead of hardcoding the value of 239 and 2 in the code, define them as constants with descriptive names.

Avoid using redundant comments: Some of the comments in the code are redundant and do not provide any additional information. Remove such comments 
to make the code more concise and readable.

Use const when possible: Declare variables that will not be modified as const to prevent accidental modification and improve readability.

Simplify the logic: The code has several if-else blocks that can be simplified. For example, instead of checking if i is equal to -1 and 
then checking if j is equal to -1, you can combine them into a single if statement that checks if either i or j is equal to -1.
