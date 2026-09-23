# Computer Networks Basics

A computer network connects devices so they can exchange data. The
internet is simply a huge network of networks, all speaking the same
common set of protocols.

## IP Addresses

Every device on a network needs an address so other devices know where
to send data, similar to a postal address for a house. This is called
an IP address. IPv4 addresses look like `192.168.1.10`, four numbers
separated by dots, while newer IPv6 addresses are longer to
accommodate far more devices than IPv4 has room for. On a home
network, a router usually hands out a private IP address to each
device automatically.

## TCP and UDP

TCP and UDP are two ways that data gets carried across a network. TCP
checks that every piece of data arrives, and in the right order,
resending anything that gets lost along the way; this reliability
makes it the right choice for things like loading a web page or
sending a file, where missing data would break something. UDP sends
data without that checking, so it is faster and has less overhead, but
some pieces can arrive out of order or not arrive at all. Video calls
and online games often use UDP, since a dropped frame here and there
matters less than the delay caused by waiting for TCP to fix it.

## Ports

A port is a number that identifies a specific service running on a
device, so that one machine can run many network services at once
without them interfering with each other. A web server usually listens
on port 80 for regular traffic and port 443 for encrypted traffic. When
your browser connects to a website, it is really connecting to a
specific IP address and port combination, even though you never type
the port number yourself.

## DNS

Typing an IP address for every website would be unworkable, so DNS
(Domain Name System) translates human-friendly names like
`example.com` into the IP address the computer actually needs to
connect to. This lookup happens automatically in the background nearly
every time you visit a site. DNS is often compared to a phonebook,
converting a name into a number needed to actually reach someone.

## HTTP

HTTP is the protocol that web browsers and servers use to talk to each
other: the browser sends a request for a page or resource, and the
server sends back a response, usually built from HTML, CSS,
JavaScript, or JSON. HTTPS is the same idea but encrypted, so that
anyone intercepting the traffic between browser and server cannot read
or tamper with it, which matters a lot when passwords or payment
details are involved.

## Routers

A router is a device that forwards data between networks, deciding the
best path for each piece of data to take on its way to its
destination. A home router connects the devices in a house to the
wider internet, while routers inside a large network can pass data
through many hops before it reaches where it's going.
