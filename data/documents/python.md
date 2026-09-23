# Python Basics

Python is an interpreted, high-level programming language known for
readable syntax. Unlike languages such as Java or C++, Python does not
require you to declare the type of a variable, and blocks of code are
defined by indentation rather than curly braces.

## Variables

A variable is a name that refers to a value stored in memory. You create
one simply by assigning a value to it:

```python
age = 21
name = "Alice"
is_student = True
```

Python figures out the type automatically based on the value assigned.
The same variable name can later be reassigned to a value of a different
type, since the name itself carries no fixed type.

## Lists and Tuples

A list is an ordered, changeable collection of items, written with
square brackets:

```python
fruits = ["apple", "banana", "cherry"]
fruits.append("orange")
```

Items in a list can be added, removed, or changed after the list is
created. A tuple looks similar but uses parentheses and cannot be
modified once created:

```python
point = (3, 4)
```

Tuples are useful when you want to make sure a group of values stays
fixed, for example coordinates or a date made of (year, month, day).

## Functions

A function lets you group a block of code under a name so you can run
it again later without retyping it. Functions are defined with the
`def` keyword:

```python
def greet(name):
    return f"Hello, {name}!"
```

Once defined, `greet("Alice")` runs that block with `name` set to
`"Alice"`. Functions can take any number of arguments and can return a
value using the `return` keyword. If a function has no `return`
statement, it returns `None` by default.

## Indentation

Python uses indentation, usually four spaces, to mark where a block of
code starts and ends. This replaces the `{ }` braces used in many other
languages. Mixing tabs and spaces, or indenting inconsistently, causes
an `IndentationError`.

```python
if age >= 18:
    print("You are an adult")
else:
    print("You are a minor")
```

## Exceptions

Things go wrong while a program runs: a file might not exist, or code
might try to divide by zero. Python reports these problems by raising
an exception, which stops the program unless it is handled.

```python
try:
    number = int(input("Enter a number: "))
    result = 10 / number
except ValueError:
    print("That wasn't a number")
except ZeroDivisionError:
    print("Cannot divide by zero")
```

A common one is `IndexError`, raised when you try to access a list
position that does not exist, such as `fruits[10]` on a three-item
list. Wrapping risky code in a `try`/`except` block lets the program
recover instead of crashing.
