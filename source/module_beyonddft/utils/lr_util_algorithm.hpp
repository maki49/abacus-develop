namespace LR_Util
{
    template <typename T>
    void delete_p2(T** p2, size_t size)
    {
        if (p2 != nullptr)
        {
            for (size_t i = 0; i < size; ++i)
            {
                if(p2[i] != nullptr) delete[] p2[i];
            }
            delete[] p2;
        }
    };

    template <typename T>
    void delete_p3(T*** p3, size_t size1, size_t size2)
    {
        if (p3 != nullptr)
        {
            for (size_t i = 0; i < size1; ++i)
            {
                delete_p2(p3[i], size2);
            }
            delete[] p3;
        }
    };
 
}