#include <cublas_v2.h>
#include "cublas_assert.hpp"

struct cublas_handle_initializer
{
	cublasHandle_t handle;

	cublas_handle_initializer()
	{
		cublas_assert( cublasCreate(&handle) );
	}

	~cublas_handle_initializer()
	{
		cublas_assert( cublasDestroy(handle) );
	}
};

template< typename T >
struct singleton
{
	typedef T value_type;
	typedef singleton self_type;
private:
	singleton( const self_type& );
	self_type& operator = ( const self_type& );
	singleton();

private:
	struct constuctor
	{
		constuctor() { self_type::instance(); }
		inline void null_action() const { }
	};

	static constuctor constuctor_;

public:
	static value_type&
		instance()
	{
		static value_type instance_;
		constuctor_.null_action();
		return instance_;
	}
};

template<typename T>
typename singleton<T>::constuctor singleton<T>::constuctor_;  