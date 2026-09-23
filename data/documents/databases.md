# Databases Basics

A database is an organized collection of data that can be stored,
searched, and updated efficiently. Most applications use a relational
database, where data is organized into tables rather than loose files.

## Tables, Rows, and Columns

A table stores data in rows and columns, similar to a spreadsheet. Each
column has a name and a data type, such as text, integer, or date. Each
row is one record. For example, a `students` table might have columns
`id`, `name`, and `age`, and each row would represent one student.

## Primary Keys

Every table usually has a primary key: a column, or combination of
columns, whose value uniquely identifies each row. No two rows can
share the same primary key value, and it cannot be empty. An `id`
column that increases automatically for every new row is a common
choice for a primary key, because it guarantees uniqueness without the
application having to think about it.

## Foreign Keys

A foreign key is a column in one table that points to the primary key
of another table, creating a link between the two. For example, an
`orders` table might have a `customer_id` column that refers to the
`id` column in a `customers` table. This is how relational databases
represent relationships between different kinds of data without
duplicating everything into one giant table.

## SQL

SQL (Structured Query Language) is the language used to read and
change data in a relational database. A few common statements:

```sql
SELECT name, age FROM students WHERE age > 18;
INSERT INTO students (name, age) VALUES ('Sam', 20);
UPDATE students SET age = 21 WHERE name = 'Sam';
DELETE FROM students WHERE name = 'Sam';
```

`SELECT` retrieves data, `INSERT` adds a new row, `UPDATE` changes
existing rows, and `DELETE` removes rows. Joining two tables together,
using a shared key, lets a single query pull related data from both at
once instead of running separate lookups by hand.

## Normalization

Normalization is the process of organizing a database so that
information is not needlessly repeated across rows and tables. If a
customer's address were copied into every single order row, updating
that address would mean fixing it in many places at once, and it would
be easy for the copies to drift out of sync. Instead, the address is
kept once in a `customers` table, and orders simply reference the
customer by id. Splitting data this way to avoid duplication is the
main idea behind normalization, and it is one of the most important
habits to build when designing a schema.

## Indexes

An index is a separate structure that a database keeps alongside a
table so it can find matching rows faster, similar to the index at the
back of a book instead of reading every page. Searching a large table
by a column without an index means scanning every row one by one,
which gets slow as the table grows. The tradeoff is that indexes take
up extra storage and slow down writes slightly, since the index has to
be updated too whenever data changes.
