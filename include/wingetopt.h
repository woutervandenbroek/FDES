///*
//   POSIX getopt for Windows
//
//   AT&T Public License
//
//   Code given out at the 1985 UNIFORUM conference in Dallas.
// */
//
//
//extern int opterr;
//extern int optind;
//extern int optopt;
//extern char *optarg;
//extern int getopt(int argc, char **argv, char *opts);
extern int getopt(int argc, char ** argv, const char* optstring);
extern int getopt_long( int argc, char ** argv, const char * optstring, const struct option * longoption, int * longindex );
struct option {
        const char * name;
        int has_arg;
        int * flag;
        int val;
};

int getopt( int argc, char ** argv, const char * optlist );
int getopt_long( int argc, char ** argv, const char * optlist, const struct option * longoption, int * longindex );

extern int optind;
extern int optopt;
extern int opterr;
extern const char * optarg;