# pgTAP

pgTAP is a suite of database functions that make it easy to write [TAP][1]-emitting unit tests in
`psql` scripts or xUnit-style test functions. The TAP output is suitable for harvesting, analysis,
and reporting by a TAP harness, such as those used in Perl applications.

Why would you want to unit test your database? Well, there are a couple of scenarios in which it can
be useful.

## Application Development

So you write PostgreSQL-backed applications, maybe in [Rails][2], or [Django][3], or [Catalyst][4],
and because you’re an [agile developer][5], you write lots of tests to make sure that your
application works as you practice iterative development. But, as one of the most important parts of
your application, should you not also test the database? Sure, you write tests of your API, and the
API covers the database, but that’s not really unit testing the database itself, is it?

pgTAP allows you to *really* test the database, not only verifying the structure of your schema, but
also by exercising any views, procedures, functions, rules, or triggers you write. Of course you
could use your application’s unit-testing framework to test the database, but by writing your tests
with pgTAP, you can keep your database tests simple. Consider these simple tests written with
[Test::More][6] and the Perl [DBI][7] to test a custom database function:

`use Test::More tests => 3;
use DBI;

my $dbh = DBI->connect('dbi:Pg:dbname=try', 'postgres', '' );

# Start a transaction.
$dbh->begin;
END { $dbh->rollback; $dbh->disconnect; }
my $domain_id = 1;
my $src_id = 2;

# Insert stuff.
ok $dbh->do(
    'SELECT insert_stuff( ?, ?, ?, ?)',
    undef, 'www.foo.com', [1, 2, 3], $domain_id, $src_id
), 'Inserting some stuff should return true;

# Grab the stuff records.
ok my $stuff = $dbh->selectall_arrayref(q{
    SELECT stuff_id
      FROM domain_stuff
     WHERE domain_id = ?
       AND src_id    = ?
     ORDER BY stuff_id
}, undef, $domain_id, $src_id), 'Fetch the domain stuff';

# Make sure we have the right stuff.
is_deeply $stuff, [ 1, 2, 3 ], 'The rows should have the right stuff';`

The upshot is that you have to connect to the database, set up transactions, execute the database
functions, fetch back data into Perl data structures, and then compare values. Now consider the
equivalent written with pgTAP:

`-- Start a transaction.
BEGIN;
SELECT plan( 2 );
\set domain_id 1
\set src_id 1

-- Insert stuff.
SELECT ok(
    insert_stuff( 'www.foo.com', '{1,2,3}', :domain_id, :src_id ),
    'insert_stuff() should return true'
);

-- Check for domain stuff records.
SELECT is(
    ARRAY(
        SELECT stuff_id
          FROM domain_stuff
         WHERE domain_id = :domain_id
           AND src_id = :src_id
         ORDER BY stuff_id
    ),
    ARRAY[ 1, 2, 3 ],
    'The stuff should have been associated with the domain'
);

SELECT * FROM finish();
ROLLBACK;`

Now isn’t that a lot easier to read? Unlike the Perl tests, the pgTAP tests can just compare values
directly in the database. There is no need to do any extra work to get the database interface to
talk to the database, fetch data, convert it, etc. You just use SQL. And if you’re working hard to
keep SQL in the database and application code in the application, why would you write database tests
in Application code? Just write them in SQL and be done with it!

## Schema Validation

Even better is the scenario in which you need to test your database schema objects, to make sure
that everything is where it should be. pgTAP provides a wealth of test functions that make schema
testing a snap!:

`BEGIN;
SELECT plan( 18 );

SELECT has_table( 'domains' );
SELECT has_table( 'stuff' );
SELECT has_table( 'sources' );
SELECT has_table( 'domain_stuff' );

SELECT has_column( 'domains', 'id' );
SELECT col_is_pk(  'domains', 'id' );
SELECT has_column( 'domains', 'domain' );

SELECT has_column( 'stuff',   'id' );
SELECT col_is_pk(  'stuff', 'id' );
SELECT has_column( 'stuff',   'name' );

SELECT has_column( 'sources', 'id' );
SELECT col_is_pk(  'sources', 'id' );
SELECT has_column( 'sources', 'name' );

SELECT has_column( 'domain_stuff', 'domain_id' );
SELECT has_column( 'domain_stuff', 'source_id' );
SELECT has_column( 'domain_stuff', 'stuff_id' );
SELECT col_is_pk(
    'domain_stuff',
    ARRAY['domain_id', 'source_id', 'stuff_id']
);

SELECT can_ok(
    'insert_stuff',
    ARRAY[ 'text', 'integer[]', 'integer', 'integer' ]
);

SELECT * FROM finish();
ROLLBACK;`

And there are many more testing functions to be had. Read the [complete documentation][8] for all
the good stuff.

## xUnit-Style Testing

In addition to the scripting-style of unit testing typical of TAP test frameworks, pgTAP also
supports xUnit-style testing of the database, similar to the approach taken by [PGUnit][9] and
[Epic][10]. It’s simple to use: Just write your test functions in the schema of your choice and use
the the pgTAP assertion functions to do the tests:

`CREATE OR REPLACE FUNCTION mytest.testschema()
RETURNS SETOF TEXT LANGUAGE plpgsql AS $$
BEGIN
    RETURN NEXT has_table( 'domains' );
    RETURN NEXT has_table( 'stuff' );
    RETURN NEXT has_table( 'sources' );
    RETURN NEXT has_table( 'domain_stuff' );
END;
$$;`

Once you’ve created your test functions and installed them in your database, you can run them at any
time using the `runtests()` function:

`SELECT * FROM runtests('mytest'::name);`

And that’s it. The `runtests()` function will handle all the particulars, including rolling back any
changes made to the schema during the running of each test. It also supports setup and teardown
functions, as well as startup and shutdown. Consult the [complete documentation][11] for details.

## Module Development

If you’re developing third-party modules for PostgreSQL, such as [custom data types][12] or [foreign
data wrappers][13], you can of course use PostgreSQL’s standard regression test architecture. But if
you’re doing agile development, writing tests as you go, which test output would you rather read?
This:

`% psql -Xd try -f pg.sql
CREATE TEMP TABLE srt (
   name CITEXT
);
CREATE TABLE
INSERT INTO srt (name)
VALUES ('aardvark'),
       ('AAA'),
       ('aba'),
       ('ABC'),
       ('abd');
INSERT 0 5
SELECT LOWER(name) as aaa FROM srt WHERE name = 'AAA'::text;
 aaa 
-----
 aaa
(1 row)
SELECT LOWER(name) as aaa FROM srt WHERE name = 'AAA'::varchar;
 aaa 
-----
 aaa
(1 row)

SELECT LOWER(name) as aaa FROM srt WHERE name = 'AAA'::bpchar;
 aaa 
-----
 aaa
(1 row)

SELECT LOWER(name) as aaa FROM srt WHERE name = 'AAA';
 aaa 
-----
 aaa
(1 row)`

Which tests pass? Which fail? Not only is it hard to tell, but because `pg_regress` compares the
output against an expected output file, you have to maintain that file, too. Too much work!

In contrast, the output for the equivalent tests using pgTAP looks like this:

`% pg_prove -d try pgtap.sql --verbose
pgtap....
1..4
ok 1 - Should find "AAA"::text
ok 2 - Should find "AAA"::varchar
ok 3 - Should find "AAA"::bpchar
ok 4 - Should find "AAA"
ok
All tests successful.
Files=1, Tests=4,  0 wallclock secs ( 0.00 usr  0.00 sys +  0.01 cusr  0.00 csys =  0.01 CPU)
Result: PASS
    `

*Now* can you tell what tests pass and fail? Can you tell what, exactly, was tested? Yeah, so can I.

## Get Started

So, what are you waiting for? [Download the latest version of pgTAP][14], or grab fork the [git
repository][15], read the [documentation][16], and get going with those tests!

[[home]][17]

* [Home][18]
* [Download][19]
* [Documentation][20]
* [pg_prove][21]
* [Integration][22]
* [Mail List][23]
* [GitHub][24]


* [Code: David E. Wheeler][25]
* [Webdesign: tri-star][26]
* [Photo: Courtland Whited][27]

[1]: http://testanything.org/
[2]: http://www.rubyonrails.org/
[3]: http://www.djangoproject.com/
[4]: http://www.catalystframework.org/
[5]: http://en.wikipedia.org/wiki/Agile_software_development
[6]: http://search.cpan.org/perldoc/Test::More
[7]: http://search.cpan.org/perldoc/DBI
[8]: documentation.html
[9]: http://en.dklab.ru/lib/dklab_pgunit/
[10]: http://www.epictest.org/
[11]: documentation.html#runtests
[12]: http://pgxn.org/tag/datatype/
[13]: http://pgxn.org/tag/fdw/
[14]: http://pgxn.org/dist/pgtap/
[15]: https://github.com/theory/pgtap/
[16]: /documentation.html
[17]: /
[18]: /
[19]: http://pgxn.org/dist/pgtap/
[20]: /documentation.html
[21]: /pg_prove.html
[22]: /integration.html
[23]: https://groups.google.com/forum/#!forum/pgtap-users
[24]: https://github.com/theory/pgtap/
[25]: http://theory.pm/
[26]: http://www.tristarwebdesign.co.uk
[27]: http://flickr.com/photos/idreaminir/
